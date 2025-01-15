#!/bin/bash
#SBATCH --job-name=pm25_predict
#SBATCH --output=logs/job_%A_%a.out
#SBATCH --error=logs/job_%A_%a.err
#SBATCH --array=0-99
#SBATCH --time=04:00:00
#SBATCH --mem=16000

# Create logs directory if it doesn't exist
mkdir -p logs

# Number of files per job
FILES_PER_JOB=2500

# Calculate start and end indices for this job
START=$((SLURM_ARRAY_TASK_ID * FILES_PER_JOB))
END=$((START + FILES_PER_JOB))

# Run the Python script
conda activate tf-gpu
python step-2-data-predict-subjob.py $START $END
