#!/bin/bash

#SBATCH --job-name=pm25_predict
#SBATCH --partition=short-serial
#SBATCH --output=logs/job_%A_%a.out
#SBATCH --error=logs/job_%A_%a.err
#SBATCH --array=0-99
#SBATCH --time=24:00:00
#SBATCH --mem=32000

# Create logs directory if it doesn't exist
mkdir -p logs-csv

python step-2.5-h5-to-csv ${SLURM_ARRAY_TASK_ID}
