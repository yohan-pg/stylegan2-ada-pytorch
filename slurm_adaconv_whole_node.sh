#!/bin/bash
#SBATCH --account=def-jlalonde
#SBATCH --nodes=1
#SBATCH --gres=gpu:4         # Number of GPU(s) per node
#SBATCH --cpus-per-task=8    # CPU cores/threads
#SBATCH --mem=32000M         # memory per node
#SBATCH --time=3-00:00       # time (DD-HH:MM)
#SBATCH --job-name=adaconv

source slurm_adaconv.sh --gpus 4