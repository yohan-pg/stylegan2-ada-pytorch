#!/bin/bash
#SBATCH --account=def-jlalonde
#SBATCH --gres=gpu:2         # Number of GPU(s) per node
#SBATCH --cpus-per-task=8    # CPU cores/threads
#SBATCH --mem=32000M         # memory per node
#SBATCH --time=3-00:00       # time (DD-HH:MM)
#SBATCH --job-name=adaconv-slowndown

source .activate

date
nvidia-smi

bash train.sh adaconv-slowdown \
    --use_adaconv True \
    "${@}"
