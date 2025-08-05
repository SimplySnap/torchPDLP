#!/bin/bash
#
#SBATCH --job-name=scipy_bench_1.1    # Name for your job
#SBATCH --output=scipy_bench_1.1%j.out   # File to save job's standard output
#SBATCH --error=scipy_bench_1.1%j.err     # File to save job's standard error
#SBATCH --time=06:00:00             # 2 hour time limit
#SBATCH --partition=debug          # The partition to run on
#SBATCH --cpus-per-task=16         # Number of CPUs/threads YES THREADS per task
#SBATCH --mem=40G                   # Memory allocation

#=========================================================================================
# Job Steps
#=========================================================================================
echo "### Starting job on host: $(hostname) at $(date)"

# 1. Load required modules (use names specific to your cluster)
module load python  # Load Python module (adjust version as needed)
module load scipy  # Load SciPy module (adjust version as needed)

# 2. Activate your Python virtual environment
source pdhg_env/bin/activate

# 3. CPU configuration
echo "### CPU Configuration:"
echo "Allocated CPUs: $SLURM_CPUS_PER_TASK"
echo "OMP_NUM_THREADS: $OMP_NUM_THREADS"
echo "Node: $SLURM_NODEID"
echo "Task ID: $SLURM_PROCID"


# Configure CPU allocation
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export OPENBLAS_NUM_THREADS=$SLURM_CPUS_PER_TASK  
export MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
export NUMEXPR_NUM_THREADS=$SLURM_CPUS_PER_TASK
export VECLIB_MAXIMUM_THREADS=$SLURM_CPUS_PER_TASK

# 4. Run diagnostics (highly recommended)
echo "### Checking CPU and PyTorch..."
lscpu | grep "Model name" -A 5

# 5. Run your python script
echo "### Executing scipy-cpu-benchmark.py..."
python -u scipy-cpu-benchmark.py

echo "### Job finished at $(date)"