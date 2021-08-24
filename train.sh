# python train.py --outdir=./training-runs --data=./datasets/afhq64cat.zip --gpus=2 --use_adaconv True --dry-run
export TORCH_EXTENSIONS_DIR=extensions
export CUDA_VISIBLE_DEVICES=1
export CXX=g++

python train.py --outdir=./training-runs --data=./datasets/afhq64cat.zip --gpus=1 