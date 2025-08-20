# torchPDLP

torchPDLP is a PyTorch-based solver for linear programs, using a restarted PDHG algorithm with enhancements for stability and performance.

## Features

- Efficient primal-dual LP solving with PyTorch acceleration
- Directory and single-file solving modes
- Outputs results and solver status in accessible formats
- Enhanced algorithmic stability and speed

## Installation

Install from PyPI:
```bash
pip install torchPDLP
```

Or install directly from GitHub:
```bash
pip install git+https://github.com/SimplySnap/torchPDLP.git@pypi-package#subdirectory=torchPDLP
```

## Usage

### 1. Solving a Directory of Problems from Command Line

You can use the provided command-line script to solve all MPS files in a directory and output a summary CSV file:

```bash
torchPDLP \
  --device gpu \
  --instance_path /path/to/mps/files \
  --tolerance 1e-4 \
  --output_path /path/to/save/results \
  --precondition \
  --fishnet \
  --primal_weight_update \
  --adaptive_stepsize \
  --max_kkt 100000
```
Argument Reference:
| Argument                 | Description                                                                  |
| ------------------------ | ---------------------------------------------------------------------------- |
| `--device`               | `'cpu'`, `'gpu'`, or `'auto'`. Uses GPU if available as default.             |
| `--instance_path`        | Path to folder with `.mps` files (default: `'feasible'`).                   |
| `--tolerance`            | Convergence tolerance (default: `1e-4`).                                     |
| `--output_path`          | Folder to save outputs and Excel results (default: `'output'`).             |
| `--precondition`         | Enable Ruiz preconditioning (optional).                                      |
| `--primal_weight_update` | Enable primal weight updates (optional).                                     |
| `--adaptive_stepsize`    | Enable adaptive step sizes for PDLP (optional).                             |
| `--infeasibility_detect` | Enable infeasibility detection (optional).                                   |
| `--fishnet`              | Use fishnet algorithm for better starting point (optional).                 |
| `--verbose`              | Enable verbose logging (optional).                                           |
| `--support_sparse`       | Use sparse matrices operations if supported (optional).                     |
| `--max_kkt`              | Maximum number of KKT passes (default: `None`).                              |
| `--time_limit`           | Time limit for the solver in seconds (default: `None`).                     |

### 2. Solving a Single Problem in Python

To solve a single MPS problem file and retrieve the solution, objective value, and solver status:

```python
import torchPDLP

result = torchPDLP.solve("path/to/file.mps")
```
torchPDLP.solve has all the same optional arguments as the command line function.
Result is a dictionary with keys:

| Key               | Description                                                                                   |
| ----------------- | --------------------------------------------------------------------------------------------- |
| `optimal_point`   | The optimal solution found by the solver (PyTorch tensor).                                    |
| `objective_value` | The value of the objective function at the optimal point (float).                             |
| `status`          | Solver status, either `"solved"` or `"unsolved"` (string).                                    |
| `time`            | Total time taken to solve the problem (in seconds, float).                                    |
| `iterations`      | Number of main algorithm iterations performed (integer).                                      |
| `restarts`        | Number of times the PDHG algorithm was restarted (integer).                                   |
| `kkt_passes`      | Number of KKT passes performed during solving (integer).                                      |


## Authors

- Xiyan Hu, Colgate University
- Titus Parker, Stanford University
- Connor Phillips, James Madison University
- Yifa Yu, University of California, Davis

## License

MIT License
