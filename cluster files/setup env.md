# Create the virtual environment
python -m venv pdhg_env

# Activate it
source pdhg_env/bin/activate

# Install packages
pip install pandas numpy openpyxl
pip install torch --index-url https://download.pytorch.org/whl/rocm6.4

# Submit a job
sbatch run_single_job.sh
