#!/bin/bash
#SBATCH --job-name=banditGRU
#SBATCH --output=logs/banditGRU.%A.%a.out
#SBATCH --error=logs/banditGRU.%A.%a.err
#SBATCH --partition=shared-cpu
#SBATCH --array=1-30
#SBATCH --mem=1000
#SBATCH --time=12:00:00

# source /home/users/f/findling/.bash_profile
# mamba activate iblenv

# extracting settings from $SLURM_ARRAY_TASK_ID
echo index $SLURM_ARRAY_TASK_ID

export PYTHONPATH="$PWD":$PYTHONPATH
# calling script

echo
python learn_to_infer.py $SLURM_ARRAY_TASK_ID