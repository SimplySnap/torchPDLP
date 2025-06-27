# PDLP-AMD-RIPS
Repo containing code for an AMD GPU implementation of Primal Dual Linear Programming (PDLP) algorithm. In collaboration with AMD and UCLA Math Dept.

This project investigates the acceleration of LP optimization on AMD GPUs using the ROCm platform and HIP to exploit fine-grained parallelism and high memory bandwidth. The core focus is the development of a robust, high-performance, open-source implementa-
tion of the Restarted Primal-Dual Hybrid Gradient (PDHG) algorithm tailored for general LP problems on AMD hardware. We evaluate its performance on standard LP test sets and real-world dataset. These gains enable near real-time optimization and establish AMD
GPUs as a competitive platform for high-performance mathematical programming. The results establish AMD GPUs as a competitive, cost-effective, and open-source platform for high-performance mathematical programming, particularly in applications where traditional solvers fall short due to latency or scalability constraints.

Our objectives include: 
1. Developing a parallel implementation of the PDLP algorithm on AMD hardware and open-source software
2. Benchmarking our implementation against CPU-based algorithms and the algorithm run
on NVIDIA hardware [Lu and Yang 2024]
3. Optimizing for AMD’s ROCm architecture
4. Testing on real-world data for energy allocation on the power grid

And Time permitting:

6. Writing a research-grade paper and submit to a journal
7. Finding more optimizations to improve PDLP algorithm itself, and implement,
hopefully providing better speedup over commercial solvers
