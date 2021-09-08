python train.py --outdir="./training-runs/$1" --kimg 5000 --data=./datasets/afhq32cat.zip --gpus=2 --metrics=fid50k,pr50k3 "${@:2}" 


