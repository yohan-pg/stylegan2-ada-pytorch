{
    set -e
    
    export CUDA_VISIBLE_DEVICES=1

    python calc_metrics.py --metrics=pr50k3_full --network=$1
    python calc_metrics.py --metrics=fid50k_full --network=$1
    python calc_metrics.py --metrics=ppl2_wend --network=$1
    
    exit
}