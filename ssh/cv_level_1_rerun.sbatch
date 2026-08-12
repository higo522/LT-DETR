#!/bin/bash
#SBATCH --account=def-ncoops
#SBATCH --job-name=cv_level_1_rerun
#SBATCH --mail-user=higo522@student.ubc.ca
#SBATCH --mail-type=END,FAIL          # one summary email for both reruns
#SBATCH --array=0-1                   # the 2 failed runs
#SBATCH --gpus-per-node=a100:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=32G
#SBATCH --time=24:00:00                
#SBATCH --output=/home/higo522/scratch/LT-DETR/logs/%x_%A_%a.out

module load StdEnv/2023 python/3.11 opencv gcc arrow/25.0.0

source ~/scratch/LT-DETR/venv/bin/activate

# Narval compute nodes have no internet access:
# wandb must run offline (sync from the login node afterwards)
export WANDB_MODE=offline
export WANDB_DIR=$SCRATCH/wandb

export LIGHTLY_TRAIN_DATALOADER_TIMEOUT_SEC=600

mkdir -p "$WANDB_DIR" "$SCRATCH/LT-DETR/logs"

cd ~/scratch/LT-DETR
python ssh/scripts/cv_level_1_narval_rerun.py "$SLURM_ARRAY_TASK_ID"