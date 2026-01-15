#!/bin/bash

# The following lines are SBATCH directives, they are read by the SLURM scheduler

#SBATCH --partition=GPU-a100
#SBATCH --gres=gpu:1      # request 1 GPU
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --time=01:00:00  # request 1 h of max runtime


# always include this, may provide useful information to the admins
echo "Running on: $SLURM_JOB_NODELIST"
# make sure uv environment is loaded
source .venv/bin/activate
echo "Starting script $1"
# run the python script ($@ represents all the arguments passed to this bash script, e.g. test.py -nmax 3)
python3 "$@"

echo "Finished."
