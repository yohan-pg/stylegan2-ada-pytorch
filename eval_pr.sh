{
    set -e
    
    python calc_metrics.py --metrics=ppl2_wend --gpus 2 --network=$1
    
    exit
}