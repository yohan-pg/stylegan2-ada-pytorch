
if [ $1 == "" ]; then
    echo "Please supply a name.";
    exit 1; 
fi

export CXX=g++
export CUDA_VISIBLE_DEVICES=0,1

# screen -S $1 -dm 
python train.py --outdir="./training-runs/$1" --data=./datasets/afhq64cat.zip --gpus=2 --kimg 5000 --use_adaconv True --metrics none --snap 10 --kimg 5000 

echo Training "'$1'" started.