#! /bin/bash

TOPIC=$1

for SEED in {0..999}; do
    if [ -e /mnt/nfs/scratch1/jpayan/predictive_expert_assignment/data/${TOPIC}.stackexchange.com/npy/alloc_rand_1_${SEED}.npy ]
    then 
        echo $SEED
    else
        echo $SEED
        sbatch \
        --job-name=compute_assts_${SEED} \
        --output=/mnt/nfs/scratch1/jpayan/logs/predictive_expert_assignment/compute_assts_${TOPIC}_${SEED}.out \
        --error=/mnt/nfs/scratch1/jpayan/logs/predictive_expert_assignment/compute_assts_${TOPIC}_${SEED}.err \
        --mem=16G \
        --nodes=1 \
        --ntasks=1 \
        --time=0-6:00:00 \
        --partition=defq \
        ./submit_compute_assts.sh ${TOPIC} $SEED
        sleep .1
    fi
done
