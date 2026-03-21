#include <iostream> 
#include <vector>
#include <chrono> 
#include <random> 
#include <algorithm>
#include <immintrin.h>
#include <thread>

using namespace std;

const int N = 1024;
using Matrix = vector<double>;

void fillMatrix(Matrix& mat){
    for(int i=0; i<(N*N); i++){
        mat[i] = (rand()%100)/10.0;
    }
}

void multiplyNaive(const Matrix& A, const Matrix& B, Matrix& C){

    for(int i=0; i<N ; i++){
        for(int j=0 ; j<N ; j++){
            double sum = 0;
            for(int k=0; k<N ; k++){
                sum += A[i*N + k]* B[k*N + j];
            }
            C[i*N + j] = sum;
        }
    }
}

void multiplyOptimized(const Matrix& A, const Matrix& B, Matrix& C){

    for(int i=0; i<N ; i++){
        for(int k=0 ; k<N ; k++){
            double temp = A[i*N + k];
            for(int j=0; j<N ; j++){
                C[i*N + j] += temp * B[k*N + j];
            }
        }
    }
}

void multiplyTiled(const Matrix& A, const Matrix& B, Matrix& C){

    int blocksize = 64;
    for(int bi = 0; bi<N ; bi+=blocksize){
        for(int bk = 0 ; bk<N ; bk+=blocksize){
            for(int bj = 0; bj<N ; bj+= blocksize){

                for(int i= bi ; i<min(bi+blocksize,N); i++){
                    for(int k = bk ; k<min(bk+blocksize,N); k++){
                        double temp = A[i*N + k];
                        for(int j = bj; j<min(bj+blocksize,N); j++){
                            C[i*N + j] += temp * B[k*N + j];
                        }
                    }
                }
            }
        }
    }
}

void multiplyAVX(const Matrix& A, const Matrix& B, Matrix& C){

    for(int i=0 ; i<N ; i++){
        for(int k=0; k<N ; k++){

            __m256d vecA = _mm256_set1_pd(A[i*N + k]);

            for(int j=0 ; j<N ; j+=4){

                __m256d vecC = _mm256_loadu_pd(&C[i*N + j]);
                __m256d vecB = _mm256_loadu_pd(&B[k*N + j]);

                vecC = _mm256_fmadd_pd(vecA,vecB,vecC);

                _mm256_storeu_pd(&C[i*N + j], vecC);
            }
        }
    }
}

void multiplyParallelAVX(const Matrix& A, const Matrix& B, Matrix& C){

    unsigned int numthreads = thread::hardware_concurrency();
    if(!numthreads) numthreads = 4;

    vector<thread> threads;
    int rowPerThread = N/ numthreads;

    for(int t=0 ; t<numthreads ; t++){

        int startRow = t*rowPerThread;
        int endRow = (t == numthreads-1) ? N:(startRow + rowPerThread);

        threads.emplace_back([&,startRow,endRow](){
            for(int i=startRow ; i<endRow ; i++){
                for(int k=0 ; k<N ; k++){

                    __m256d vecA = _mm256_set1_pd(A[i*N + k]);
                    
                    for(int j=0 ; j<N ; j+=4){

                        __m256d vecC = _mm256_loadu_pd(&C[i*N + j]);
                        __m256d vecB = _mm256_loadu_pd(&B[k*N + j]);
                        
                        vecC = _mm256_fmadd_pd(vecA, vecB, vecC);
                        _mm256_storeu_pd(&C[i*N + j], vecC);
                    }
                }
            }
        });
    }

    for(auto& t:threads){
        t.join();
    }

}


int main(){

    Matrix A(N*N);
    Matrix B(N*N);
    Matrix C(N*N, 0);

    fillMatrix(A);
    fillMatrix(B);

    cout<<"Starting Benchmark ("<< N << "x"<<N<<")...\n";
    cout<<"-------------------------------\n";

    auto start = chrono::high_resolution_clock::now();
    multiplyNaive(A,B,C);
    auto end = chrono::high_resolution_clock::now();
    chrono::duration<double> diffNaive = end-start;
    cout<< "1. Naive Time: "<<diffNaive.count()<<" seconds\n";


    fill(C.begin(),C.end(),0);
    start = chrono::high_resolution_clock::now();
    multiplyOptimized(A,B,C);
    end = chrono::high_resolution_clock::now();
    chrono::duration<double> diffOpt = end-start;
    cout<< "2. Optimized Time: "<<diffOpt.count()<<" seconds\n";


    fill(C.begin(),C.end(),0);
    start = chrono::high_resolution_clock::now();
    multiplyTiled(A,B,C);
    end = chrono::high_resolution_clock::now();
    chrono::duration<double> diffTiled = end-start;
    cout<< "3. Tiled Time: "<<diffTiled.count()<<" seconds\n";

    fill(C.begin(),C.end(),0);
    start = chrono::high_resolution_clock::now();
    multiplyAVX(A,B,C);
    end = chrono::high_resolution_clock::now();
    chrono::duration<double> diffAVX = end-start;
    cout<< "4. AVX Time: "<<diffAVX.count()<<" seconds\n";

    fill(C.begin(),C.end(),0);
    start = chrono::high_resolution_clock::now();
    multiplyParallelAVX(A,B,C);
    end = chrono::high_resolution_clock::now();
    chrono::duration<double> diffPar = end-start;
    cout<< "5. Parallel AVX Time: "<<diffPar.count()<<" seconds\n";


    cout<< "--------------------------------\n";
    cout<<"Speedup: "<<diffNaive.count()/diffPar.count()<<"x FASTER\n";
    cout<< "--------------------------------\n";


    return 0;

}
