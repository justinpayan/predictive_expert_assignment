#! /bin/bash

for SEED in {0..999}; do
    if [ -e /mnt/nfs/scratch1/jpayan/predictive_expert_assignment/data/cs.stackexchange.com/npy/alloc_${SEED}.npy ]
    then 
        echo $SEED
    else
        echo $SEED
        sbatch \
        --job-name=compute_assts_${SEED} \
        --output=/mnt/nfs/scratch1/jpayan/logs/predictive_expert_assignment/compute_assts_${SEED}.out \
        --error=/mnt/nfs/scratch1/jpayan/logs/predictive_expert_assignment/compute_assts_${SEED}.err \
        --nodes=1 \
        --ntasks=1 \
        --time=0-6:00:00 \
        --partition=defq \
        ./submit_compute_assts.sh $SEED
        sleep 1
    fi
done
