# * Note that on tch, the device # appears reversed (titan v is 1, titant rtx is 0)

export CXX=g++
export CUDA_VISIBLE_DEVICES=1

python train.py --cfg custom --outdir="./training-runs/$1" --kimg 5000 --data=./datasets/afhq64cat.zip --metrics none  --gpus=1 --snap 10
