#!/bin/bash
#SBATCH --account=def-ncoops
#SBATCH --job-name=cv_level_2
#SBATCH --mail-user=higo522@student.ubc.ca
#SBATCH --mail-type=END,FAIL          # one summary email for the whole array
#SBATCH --array=0-19                  # 5 test folds x 4 heldout splits = 20 runs, one per node
#SBATCH --gpus-per-node=a100:1
#SBATCH --cpus-per-task=12            # Narval A100 nodes: 12 cores per GPU
#SBATCH --mem=64G
#SBATCH --time=12:00:00                # adjust after timing one fold
#SBATCH --output=/home/higo522/scratch/LT-DETR/logs/%x_%A_%a.out

module load StdEnv/2023 python/3.11 opencv

source ~/scratch/LT-DETR/venv/bin/activate

# Narval compute nodes have no internet access:
# wandb must run offline (sync from the login node afterwards)
export WANDB_MODE=offline
export WANDB_DIR=$SCRATCH/wandb

# pretrained DINOv3 backbone must already sit here as dinov3_convnext_small_lvd1689m.pth (transferred via Globus)
export LIGHTLY_TRAIN_MODEL_CACHE_DIR=$SCRATCH/LT-DETR/model_cache

mkdir -p "$WANDB_DIR" "$SCRATCH/LT-DETR/logs" "$LIGHTLY_TRAIN_MODEL_CACHE_DIR"

cd ~/scratch/LT-DETR
python ssh/scripts/cv_level_2_narval.py "$SLURM_ARRAY_TASK_ID"
