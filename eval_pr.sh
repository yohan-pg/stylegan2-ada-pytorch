{
    set -e
    
    python calc_metrics.py --metrics=avg_min_dist --gpus 2 --network=$1
    
    exit
}