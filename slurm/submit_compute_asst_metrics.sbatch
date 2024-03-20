#! /bin/bash

for SEED in {0..999}; do
    if [ -e /mnt/nfs/scratch1/jpayan/predictive_expert_assignment/data/cs.stackexchange.com/npy/metric_to_allocation_scores_${SEED}.pkl ]
    then 
        echo $SEED
    else
        sbatch \
        --job-name=compute_asst_metrics_${SEED} \
        --output=/mnt/nfs/scratch1/jpayan/logs/predictive_expert_assignment/compute_asst_metrics_${SEED}.out \
        --error=/mnt/nfs/scratch1/jpayan/logs/predictive_expert_assignment/compute_asst_metrics_${SEED}.err \
        --nodes=1 \
        --ntasks=1 \
        --time=0-6:00:00 \
        --partition=defq \
        ./submit_compute_asst_metrics.sh $SEED
        sleep 1
    fi
done
