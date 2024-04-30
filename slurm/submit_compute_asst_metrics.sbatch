#! /bin/bash

TOPIC=$1

for SEED in {0..999}; do
    if [ -e /mnt/nfs/scratch1/jpayan/predictive_expert_assignment/data/${TOPIC}.stackexchange.com/npy/metric_to_allocation_scores_${SEED}.pkl ]
    then 
        echo $SEED
    else
        sbatch \
        --job-name=compute_asst_metrics_${SEED} \
        --output=/mnt/nfs/scratch1/jpayan/logs/predictive_expert_assignment/compute_asst_metrics_${TOPIC}_${SEED}.out \
        --error=/mnt/nfs/scratch1/jpayan/logs/predictive_expert_assignment/compute_asst_metrics_${TOPIC}_${SEED}.err \
        --nodes=1 \
        --ntasks=1 \
        --time=0-6:00:00 \
        --partition=defq \
        ./submit_compute_asst_metrics.sh $TOPIC $SEED
        sleep 1
    fi
done
