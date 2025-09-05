# torchPDLP
Repo containing code for an AMD GPU implementation of Primal Dual Linear Programming (PDLP) algorithm. In collaboration with AMD and Institute for Pure and Applied Mathematics @UCLA

This project investigates the acceleration of linear programming on AMD GPUs using the ROCm to exploit parallelism and high memory bandwidth. The core focus is the development of a robust, high-performance, open-source implementa-
tion of the Restarted Primal-Dual Hybrid Gradient (PDHG) algorithm tailored for general LP problems on AMD hardware using PyTorch. 

We evaluate its performance on standard LP test sets and real-world dataset. These gains enable near real-time optimization and demonstrate that both AMD GPUs, and PyTorch, are platforms that can competitively solve LP problems, especially when compared to traditional (barrier or simplex based) methods.

The 'master' branch contains the necessary code for running torchPDLP. If pulling or forking, please use torchPDLP. The 'main' branch, however, contains more auxillary files, and was the main branch we used for ideating. See main for commit history, jupyter notebooks, etc.

**For instructions on how to use, go to the PDLP folder - you can run from console when pulling, or import the package (see py-pi branch for instructions)**. Thank you!
