if [ $1 == "" ]; then
    echo "Please supply a name.";
    exit 1; 
fi

# * Note that on tch, the device # appears reversed (titan v is 1, titant rtx is 0)

python train.py --cfg custom --outdir="./training-runs/$1" --kimg 5000 --data=./datasets/afhq64cat.zip --metrics none  --gpus=2 --snap 10
