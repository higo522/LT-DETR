#!/bin/bash
#SBATCH --account=def-ncoops
#SBATCH --job-name=eval
#SBATCH --mail-user=higo522@student.ubc.ca
#SBATCH --mail-type=END,FAIL
#SBATCH --gpus-per-node=a100:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=/home/higo522/scratch/LT-DETR/logs/%x_%j.out

module load StdEnv/2023 python/3.11 opencv gcc arrow/25.0.0

source ~/scratch/LT-DETR/venv/bin/activate

cd ~/scratch/LT-DETR
python ssh/scripts/eval.py