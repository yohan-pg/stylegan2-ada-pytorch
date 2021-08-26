#!!! recall that batch size is 16 instead of 32 in config

if [ $1 == "" ]; then
    echo "Please supply a name.";
    exit 1; 
fi

# screen -S $1 -dm 
python train.py --outdir="./training-runs/$1" --data=./datasets/afhq32cat.zip --gpus=1  --kimg 5000 --use_adaconv True --metrics none --snap 10 

echo Training "'$1'" started.