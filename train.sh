#!!! recall that batch size is 16 instead of 32 in config

python train.py --outdir="./training-runs/$1" --data=./datasets/afhq32cat.zip --gpus=1 --kimg 5000 --metrics none --snap 10 