#!/bin/bash

#SBATCH --partition=orchid
#SBATCH --account=orchid
#SBATCH --gres=gpu:1
#SBATCH --job-name=training_simple
#SBATCH -o %A.out
#SBATCH -e %A.err
#SBATCH --time=24:00:00
#SBATCH --mem=48000

# Your script execution
conda activate tf-gpu
python ml_train.py