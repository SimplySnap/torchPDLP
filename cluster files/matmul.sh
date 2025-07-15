#!/bin/bash
#SBATCH -J matmul-py         #Job name
#SBATCH -o %x.%J.out         #Job output file
#SBATCH --error=%J.err       # File to save job's standard erro
#SBATCH -p debug             # -p sets the partition node - here is debug
#SBATCH --gpus=mi210:1 -N1 -n1 -c1         # 1 GPU, 1 node, 1 task, 1 cpu
#SBATCH -t 00:30:00           # Time 30 mins
#SBATCH --mem=32G            #We typically need quite a lot of memory lol - our max is 80G
#memory blowup is real, esp with matrix multiplication.... #justsaying :3


set -x #enables verbose output
hostname #prints name of compute node we're running on
rocm-smi #ROCM command that shows diagnostic info about hardware

rocminfo | grep "Device Type" -A 5

module load py-torch/2.5.1-hip6.2.4 #Load torch rn

source pdhg_env/bin/activate #activate environment

# Run your Python script
python -u 'matmul.py'
