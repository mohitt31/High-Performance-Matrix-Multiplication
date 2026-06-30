# ⚡ High-Performance Matrix Multiplication Engine (HPC)

![C++](https://img.shields.io/badge/Standard-C++17-blue?style=for-the-badge&logo=c%2B%2B)
![Optimization](https://img.shields.io/badge/Optimization-Manual%20AVX2-orange?style=for-the-badge)
![Technique](https://img.shields.io/badge/Technique-Cache%20Blocking-green?style=for-the-badge)
![Build](https://img.shields.io/badge/Build-Passing-brightgreen?style=for-the-badge)

> **"Optimizing beyond the Compiler's limits."**

This project implements a highly optimized Matrix Multiplication engine. Unlike standard implementations that rely on compiler auto-vectorization, this project utilizes **Manual SIMD Intrinsics (AVX2)** and **Multithreading** to squeeze every bit of performance from the CPU.

Achieved an **75x Speedup** natively on Linux x86_64 hardware with strictly verified correctness checks.

---

## 🏎️ Performance Benchmarks (1024x1024)

Benchmarks recorded natively on **Linux x86_64 (Intel Xeon Platinum 8272CL CPU @ 2.60GHz)**.
*(Note: Median execution time over 5 runs with strict max absolute difference correctness checking)*

| Optimization Level | Median Time | Speedup | Technical Breakdown |
| :--- | :--- | :--- | :--- |
| **1. Naive** | `4.471 s` | 1.0x | Baseline $O(N^3)$ algorithm. Heavy cache misses. |
| **2. Optimized** | `0.301 s` | 14.83x | **Loop Reordering (`i-k-j`)**: Maximizes Spatial Locality. |
| **3. Tiled** | `0.231 s` | 19.34x | **L1 Cache Blocking**: 64x64 tiles to prevent Cache Thrashing. |
| **4. AVX2 (Manual)** | `0.219 s` | 20.34x | **Explicit Vectorization**: Using `_mm256_fmadd_pd` manually. |
| **5. Parallel AVX** | **0.059 s** | **75.17x** | **Multithreading**: `std::thread` pool with AVX2 kernels. |

### 🔍 Architectural Findings: Memory-Bound AVX Loops vs GCC Auto-Vectorization
During rigorous benchmarking against AMD EPYC and Intel Xeon environments, an anomaly was discovered: the `Optimized` (auto-vectorized GCC) kernel initially outperformed the manually-unrolled `AVX2` intrinsics. Analysis revealed that the naive `i-k-j` order inside the AVX loop forced the compiler to load and store the `C` matrix on every single element of `k` — a catastrophic memory bottleneck. 

This repository was subsequently updated to use an `i-j-k` accumulation order for the manual intrinsics path with `64x64` L1 cache blocking, allowing the kernel to load `C` only once into the `YMM` register, accumulate across `k`, and store `C` once. With this fix, the manual AVX implementations now natively dominate the execution profile, achieving up to 75x speedup with ThreadSanitizer-verified, lock-free parallel scaling.

### 📊 Visual Analysis
![Benchmark Graph](benchmark_graph.png)

---

## 🧠 Why This Project is Different?

Most optimizations rely on the compiler to "Auto-Vectorize" code. This project goes a step further by using **Hardware Intrinsics**:

### 1. Manual AVX2 Implementation (`immintrin.h`)
Instead of hoping the compiler optimizes the math, I explicitly mapped data to **256-bit YMM Registers**.
* **Instruction:** `_mm256_fmadd_pd` (Fused Multiply-Add).
* **Throughput:** Processes **4 Double-Precision** numbers in a single CPU cycle.

### 2. Cache-Aware Tiling
Implemented **Block Matrix Multiplication** with a tile size of `64`. This ensures that the working set fits entirely inside the **L1 Cache** (32KB-64KB), reducing expensive RAM fetches by order of magnitude.

### 3. Lock-Free Parallelism
Used `std::thread` with a lambda-based worker model. The matrix is sliced row-wise (`startRow` to `endRow`), ensuring **Zero Race Conditions** without needing Mutex locks (which slow down performance).

---

## 💻 How to Run

### Prerequisites
* **CPU:** Intel/AMD with AVX2 Support.
* **Compiler:** GCC (g++) or Clang.
* **OS:** Linux/WSL.

### Build & Benchmark
```bash
# Compile with O3 optimizations, AVX2 flags, and pthread linking
g++ -O3 -mavx2 -mfma -pthread main.cpp -o matrix

# Run the engine
./matrix

---

## 👨‍💻 Author

*Mohit Prajapati*
*High-Performance Computing & Systems Engineering Enthusiast*