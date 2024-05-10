#! /bin/bash

TOPIC=$1

sbatch \
--job-name=${TOPIC}_emb_sims \
--output=/work/pi_yzick_umass_edu/jpayan/logs/predictive_expert_assignment/emb_sims_${TOPIC}.out \
--error=/work/pi_yzick_umass_edu/jpayan/logs/predictive_expert_assignment/emb_sims_${TOPIC}.err \
--nodes=1 \
--ntasks=1 \
--mem=128G \
--time=0-10:00:00 \
--partition=cpu \
./submit_compute_emb_sims.sh $TOPIC

