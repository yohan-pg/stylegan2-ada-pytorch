{
    set -e
    
    # python calc_metrics.py --metrics=fid50k_full --gpus 2 --network=$1 "${@:2}"
    # python calc_metrics.py --metrics=ppl2_wend --gpus 2 --network=$1 "${@:2}"
    python calc_metrics.py --metrics=pr50k3_full --gpus 2 --network=$1 "${@:2}"
    
    exit
}