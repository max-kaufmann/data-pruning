#!/bin/bash
#SBATCH --nodes=1
#SBATCH --time 0-16:00:00

date;hostname;id;pwd
source ~/.bashrc
export WANDB_API_KEY= '5e79ee5b62c4ec69428d4db62bd114ef2e4df187'
conda activate base
source /opt/rh/devtoolset-10/enable

echo "Starting job $SLURM_ARRAY_JOB_ID.$SLURM_ARRAY_TASK_ID"
train_script=~/data-pruning/scripts/run_train.py

echo $train_script
python --version
python $train_script --file $2 --job_id $SLURM_ARRAY_JOB_ID --task_id $SLURM_ARRAY_TASK_ID 