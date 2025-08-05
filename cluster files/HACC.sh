#!/bin/bash
#
#SBATCH --job-name=small_lp_pdlp     # Name for your job
#SBATCH --output=small_lp_pdlp_%j.out   # File to save job's standard output
#SBATCH --error=small_lp_pdlp%j.err     # File to save job's standard error
#SBATCH --partition=debug          # The partition to run on
#SBATCH --cpus-per-task=4          # Number of CPUs per task
#SBATCH --mem=80G                   # Memory allocation
#SBATCH --gpus=mi210:1             # Request 1 MI210 GPU

#=========================================================================================
# Job Steps
#=========================================================================================
echo "### Starting job on host: $(hostname) at $(date)"

# 1. Load required modules (use names specific to your cluster)
module load py-torch/2.5.1-hip6.2.4

# 2. Activate your Python virtual environment
source pdhg_env/bin/activate

# 3. Run diagnostics (highly recommended)
echo "### Checking GPU and PyTorch..."
rocminfo | grep "Device Type" -A 5
python -c "import torch; print(f'PyTorch CUDA available: {torch.cuda.is_available()}'); print(f'Device: {torch.cuda.get_device_name(0)}')"

# 4. Run your python script
echo "### Executing main.py..."
python -u PDLP/main.py

echo "### Job finished at $(date)"