if [ $1 == "" ]; then
    echo "Please supply a name.";
    exit 1; 
fi

python train.py --outdir="./training-runs/$1" --kimg 5000 --data=./datasets/afhq32cat.zip --gpus=2 --metrics=fid50k,pr50k3 --use_adaconv True "${@:2}"

# --metrics none --snap 10 
