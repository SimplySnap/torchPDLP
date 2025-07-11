#!/bin/bash

#SBATCH --job-name=pdhg-single         # Job name
#SBATCH --output=job.%j.out            # Name of stdout output file (%j expands to jobId)
#SBATCH --error=job.%j.err             # Name of stderr output file
#SBATCH --nodes=1                      # Total number of nodes requested
#SBATCH --ntasks-per-node=1            # A single task for our python script
#SBATCH --partition=mi2104x
#SBATCH --time=24:00:00                # Run time limit of 30 minutes for the devel queue

#=========================================================================================
# Job Steps
#=========================================================================================
echo "### Starting job on host: $(hostname) at $(date)"

# 1. Load required modules (use names specific to your cluster)
module load rocm pytorch

# 2. Activate your Python virtual environment
source pdhg_env/bin/activate

# 3. Run diagnostics (highly recommended)
echo "### Checking GPU and PyTorch..."
rocminfo | grep "Device Type" -A 5
python -c "import torch; print(f'PyTorch CUDA available: {torch.cuda.is_available()}'); print(f'Device: {torch.cuda.get_device_name(0)}')"

# 4. Run your python script
echo "### Executing netlib_gpu.py..."
python -u netlib_gpu.py

echo "### Job finished at $(date)"