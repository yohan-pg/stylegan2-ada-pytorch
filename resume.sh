
python train.py \
    --kimg 5000 \
    --data=./datasets/afhq32cat.zip \
    --metrics=fid50k,pr50k3,ppl2_wend \
    --snap 50 \
    --gpus=2 \
    --resume training-runs/noise_disabled/00001-afhq32cat-auto2-gamma10-kimg5000-batch8/network-snapshot-001000.pkl \
    --outdir="./training-runs/resume_without_noise" \
    "${@:2}"


    
