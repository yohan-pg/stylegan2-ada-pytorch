{
    set -e
    
    python calc_metrics.py --metrics=fid50k_full --gpus 1 --network=$1
    python calc_metrics.py --metrics=pr50k3_full --gpus 1 --network=$1
    python calc_metrics.py --metrics=ppl2_wend --gpus 1 --network=$1
    
    exit
}