#!/bin/bash

#SBATCH --job-name=pm25_predict
#SBATCH --partition=short-serial
#SBATCH --output=logs/job_%A_%a.out
#SBATCH --error=logs/job_%A_%a.err
#SBATCH --array=0-99
#SBATCH --time=24:00:00
#SBATCH --mem=32000

# Create logs directory if it doesn't exist
mkdir -p logs-predict

# Your script execution
conda activate tf-gpu
python step-2-data-predict-patch.py ${SLURM_ARRAY_TASK_ID}
