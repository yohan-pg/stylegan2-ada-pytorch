
export CXX=g++
export CUDA_VISIBLE_DEVICES=0

# screen -S $1 -dm 
python train.py --outdir="./training-runs/debug" --data=./datasets/afhq32cat.zip --batch 8 --gpus=1 --kimg 5000 --use_adaconv True --metrics none --snap 10 
