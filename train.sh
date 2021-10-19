
python train.py \
    --kimg 5000 \
    --data=./datasets/afhq2_cat128.zip \
    --cfg auto1 \
    --batch 8 \
    --gamma 20 \
    --gpus=2 \
    --fp32 True \
    --outdir="./training-runs/$1" \
    "${@:2}"

    # --metrics=fid50k,ppl2_wend \
    # --metrics=fid50k,pr50k3,ppl2_wend \
    # --cfg stylegan2map2 \
    # --batch 8 \
