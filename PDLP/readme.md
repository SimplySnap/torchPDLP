## How to run
```bash
python -u /path/to/main.py \
  --device gpu \
  --instance_path /path/to/mps/files \
  --tolerance 1e-2 \
  --output_path /path/to/save/results \
  --precondition \
  --primal_weight_update \
  --adaptive_stepsize
```
## Argument Reference

| Argument                 | Description                                                                  |
| ------------------------ | ---------------------------------------------------------------------------- |
| `--device`               | `'cpu'`, `'gpu'`, or `'auto'`. Uses GPU if available as default.             |
| `--instance_path`        | Path to folder with `.mps` files.                                            |
| `--tolerance`            | Convergence tolerance (default: `1e-2`).                                     |
| `--output_path`          | Folder to save outputs and Excel results.                                    |
| `--precondition`         | Enable Ruiz preconditioning (optional).                                      |
| `--primal_weight_update` | Enable primal weight updates (optional).                                     |
| `--adaptive_stepsize`    | Enable adaptive step sizes (optional).                                       |
| `--verbose`              | Enable verbose logging (optional).                                           |
| `--support_sparse`       | Use sparse matrices if supported (optional).                                 |
