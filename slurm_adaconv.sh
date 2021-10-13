#!/bin/bash
#SBATCH --account=def-jlalonde
#SBATCH --gres=gpu:1         # Number of GPU(s) per node
#SBATCH --cpus-per-task=8    # CPU cores/threads
#SBATCH --mem=32000M         # memory per node
#SBATCH --time=3-00:00       # time (DD-HH:MM)
#SBATCH --job-name=adaconv 

source .activate

data
nvidia-smi

bash train.sh \
    --use_adaconv True \
    --gpus 1 \
    --outdir="./training-runs/adaconv" \
    "${@}"
