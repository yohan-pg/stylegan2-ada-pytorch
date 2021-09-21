#!/bin/bash
#SBATCH --account=def-jlalonde
#SBATCH --gres=gpu:1         # Number of GPU(s) per node
#SBATCH --cpus-per-task=8    # CPU cores/threads
#SBATCH --mem=32000M         # memory per node
#SBATCH --time=3-00:00       # time (DD-HH:MM)
#SBATCH --job-name=baseline

nvidia-smi
source .activate

python train.py \
    --kimg 25000 \
    --data=./datasets/afhq128cat.zip \
    --cfg stylegan2 \
    --metrics=fid50k,pr50k3,ppl2_wend \
    --outdir="./training-runs/baseline" \
    "${@}"
