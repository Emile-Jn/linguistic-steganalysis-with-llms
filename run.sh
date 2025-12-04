#!/bin/bash

# The following lines are SBATCH directives, they are read by the SLURM scheduler

#SBATCH --partition=GPU-a100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --time=01:00:00  # request 1 h of max runtime


# always include this, may provide useful information to the admins
echo "Running on: $SLURM_JOB_NODELIST"
# run the python script
python3 "$1"

echo "Finished."
