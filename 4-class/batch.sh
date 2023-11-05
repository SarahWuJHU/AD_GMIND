#!/bin/bash -l
#SBATCH
#SBATCH --job-name=GMIND_CLASS_1
#SBATCH --time=2:30:0
#SBATCH -p defq
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --array=1-10
# number of cpus (threads) per task (process)

#### load and unload modules you may need

module load cuda/10.2.89
module load anaconda
conda activate gmind
python3.7 wrapper_img.py $SLURM_ARRAY_TASK_ID
