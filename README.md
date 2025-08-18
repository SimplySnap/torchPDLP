# torchPDLP
Repo containing code for an AMD GPU implementation of Primal Dual Linear Programming (PDLP) algorithm. In collaboration with AMD and Institute for Pure and Applied Mathematics @UCLA

This project investigates the acceleration of LP optimization on AMD GPUs using the ROCm platform and HIP to exploit fine-grained parallelism and high memory bandwidth. The core focus is the development of a robust, high-performance, open-source implementa-
tion of the Restarted Primal-Dual Hybrid Gradient (PDHG) algorithm tailored for general LP problems on AMD hardware. We evaluate its performance on standard LP test sets and real-world dataset. These gains enable near real-time optimization and establish AMD
GPUs as a competitive platform for high-performance mathematical programming. The results establish AMD GPUs as a competitive, cost-effective, and open-source platform for high-performance mathematical programming, particularly in applications where traditional solvers fall short due to latency or scalability constraints.

The 'master' branch contains the necessary code for running torchPDLP. If pulling or forking, please use torchPDLP. The 'main' branch, however, contains more auxillary files, and was the main branch we used for ideating. See main for commit history, jupyter notebooks, etc.
