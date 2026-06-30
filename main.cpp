#include <iostream>
#include <vector>
#include <chrono>
#include <random>
#include <algorithm>
#include <immintrin.h>
#include <thread>
#include <fstream>
#include <cmath>
#include <iomanip>

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
    int blocksize = 64;
    for(int bi = 0; bi<N ; bi+=blocksize){
        for(int bj = 0 ; bj<N ; bj+=blocksize){
            for(int bk = 0; bk<N ; bk+= blocksize){
                for(int i= bi ; i<min(bi+blocksize,N); i++){
                    for(int j= bj; j<min(bj+blocksize,N); j+=4){
                        __m256d vecC = _mm256_loadu_pd(&C[i*N + j]);
                        for(int k = bk ; k<min(bk+blocksize,N); k++){
                            __m256d vecA = _mm256_set1_pd(A[i*N + k]);
                            __m256d vecB = _mm256_loadu_pd(&B[k*N + j]);
                            vecC = _mm256_fmadd_pd(vecA, vecB, vecC);
                        }
                        _mm256_storeu_pd(&C[i*N + j], vecC);
                    }
                }
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
            int blocksize = 64;
            for(int bi = startRow; bi<endRow ; bi+=blocksize){
                for(int bj = 0 ; bj<N ; bj+=blocksize){
                    for(int bk = 0; bk<N ; bk+= blocksize){
                        for(int i= bi ; i<min(bi+blocksize,endRow); i++){
                            for(int j= bj; j<min(bj+blocksize,N); j+=4){
                                __m256d vecC = _mm256_loadu_pd(&C[i*N + j]);
                                for(int k = bk ; k<min(bk+blocksize,N); k++){
                                    __m256d vecA = _mm256_set1_pd(A[i*N + k]);
                                    __m256d vecB = _mm256_loadu_pd(&B[k*N + j]);
                                    vecC = _mm256_fmadd_pd(vecA, vecB, vecC);
                                }
                                _mm256_storeu_pd(&C[i*N + j], vecC);
                            }
                        }
                    }
                }
            }
        });
    }
    for(auto& t:threads){
        t.join();
    }
}

double checkCorrectness(const Matrix& baseline, const Matrix& test) {
    double max_diff = 0.0;
    for (size_t i = 0; i < baseline.size(); ++i) {
        double diff = std::abs(baseline[i] - test[i]);
        if (diff > max_diff) {
            max_diff = diff;
        }
    }
    return max_diff;
}

std::string getEnvironment() {
#ifdef __linux__
    std::ifstream cpuinfo("/proc/cpuinfo");
    std::string line;
    while (std::getline(cpuinfo, line)) {
        if (line.substr(0, 10) == "model name") {
            size_t pos = line.find(":");
            if (pos != std::string::npos) {
                return "Linux, CPU:" + line.substr(pos + 1);
            }
        }
    }
    return "Linux";
#elif defined(__APPLE__)
    return "macOS";
#else
    return "Unknown OS";
#endif
}

struct BenchmarkResult {
    string name;
    double median_time;
    double speedup;
    double max_diff;
    bool passed;
};

double runBenchmark(const string& name, void (*func)(const Matrix&, const Matrix&, Matrix&), 
                    const Matrix& A, const Matrix& B, Matrix& C, const Matrix* baseline, 
                    int runs, BenchmarkResult& result) {
    vector<double> times;
    cout << "Benchmarking " << name << " (5 runs)...\n";
    for(int i=0; i<runs; i++) {
        fill(C.begin(), C.end(), 0);
        auto start = chrono::high_resolution_clock::now();
        func(A, B, C);
        auto end = chrono::high_resolution_clock::now();
        times.push_back(chrono::duration<double>(end - start).count());
    }
    sort(times.begin(), times.end());
    double median = times[runs / 2];
    
    result.name = name;
    result.median_time = median;
    if (baseline) {
        result.max_diff = checkCorrectness(*baseline, C);
        double tolerance = 1e-9 * N; // Tolerance for double-precision accumulation
        result.passed = (result.max_diff <= tolerance);
    } else {
        result.max_diff = 0;
        result.passed = true;
    }
    
    return median;
}

int main(){
    Matrix A(N*N);
    Matrix B(N*N);

    fillMatrix(A);
    fillMatrix(B);

    cout<<"Starting Benchmark ("<< N << "x"<<N<<")...\n";
    cout<<"========================================================================\n";
    cout<<"Environment: " << getEnvironment() << "\n";
    cout<<"Hardware Concurrency (Threads): " << thread::hardware_concurrency() << "\n";
    cout<<"========================================================================\n\n";

    Matrix baselineC(N*N, 0);
    BenchmarkResult naiveRes, optRes, tiledRes, avxRes, parRes;
    
    int runs = 5;

    double naiveTime = runBenchmark("Naive", multiplyNaive, A, B, baselineC, nullptr, runs, naiveRes);
    naiveRes.speedup = 1.0;
    
    Matrix C(N*N, 0);
    double optTime = runBenchmark("Optimized", multiplyOptimized, A, B, C, &baselineC, runs, optRes);
    optRes.speedup = naiveTime / optTime;

    double tiledTime = runBenchmark("Tiled", multiplyTiled, A, B, C, &baselineC, runs, tiledRes);
    tiledRes.speedup = naiveTime / tiledTime;

    double avxTime = runBenchmark("AVX (Manual)", multiplyAVX, A, B, C, &baselineC, runs, avxRes);
    avxRes.speedup = naiveTime / avxTime;

    double parTime = runBenchmark("Parallel AVX", multiplyParallelAVX, A, B, C, &baselineC, runs, parRes);
    parRes.speedup = naiveTime / parTime;

    cout << "\n========================================================================\n";
    cout << left << setw(15) << "Method" 
         << setw(17) << "Median Time(s)" 
         << setw(13) << "Speedup" 
         << setw(15) << "Max Diff" 
         << "Correctness\n";
    cout << "------------------------------------------------------------------------\n";
    
    auto printRes = [](const BenchmarkResult& r) {
        cout << left << setw(15) << r.name 
             << setw(17) << fixed << setprecision(5) << r.median_time 
             << setw(12) << fixed << setprecision(2) << r.speedup << "x"
             << " "
             << setw(14) << scientific << setprecision(2) << r.max_diff
             << (r.passed ? "PASS" : "FAIL") << "\n";
    };
    
    printRes(naiveRes);
    printRes(optRes);
    printRes(tiledRes);
    printRes(avxRes);
    printRes(parRes);
    cout << "========================================================================\n";

    ofstream csv("results.csv");
    csv << "method,median_time_seconds,speedup_vs_naive,correctness_max_abs_diff,passed\n";
    auto writeCsv = [&csv](const BenchmarkResult& r) {
        csv << r.name << "," 
            << fixed << setprecision(5) << r.median_time << "," 
            << fixed << setprecision(2) << r.speedup << "," 
            << scientific << setprecision(2) << r.max_diff << "," 
            << (r.passed ? "true" : "false") << "\n";
    };
    writeCsv(naiveRes);
    writeCsv(optRes);
    writeCsv(tiledRes);
    writeCsv(avxRes);
    writeCsv(parRes);
    csv.close();
    cout << "Results written to results.csv\n";

    return 0;
}
