#!/bin/bash
#SBATCH --account=def-jlalonde
#SBATCH --gres=gpu:1         # Number of GPU(s) per node
#SBATCH --cpus-per-task=8    # CPU cores/threads
#SBATCH --mem=32000M         # memory per node
#SBATCH --time=3-00:00       # time (DD-HH:MM)
#SBATCH --job-name=adain

source .activate

date
nvidia-smi

python train.py \
    --outdir="./training-runs/adain" \
    "${@}"
