#! /bin/bash

TOPIC=$1

sbatch \
--job-name=${TOPIC}_embed_qa \
--output=/work/pi_yzick_umass_edu/jpayan/logs/predictive_expert_assignment/embed_qa_${TOPIC}.out \
--error=/work/pi_yzick_umass_edu/jpayan/logs/predictive_expert_assignment/embed_qa_${TOPIC}.err \
--nodes=1 \
--ntasks=1 \
--mem-per-gpu=32G \
-G 1 \
--time=3-00:00:00 \
--partition=gpu-preempt \
--constraint=v100 \
./submit_embed_qa.sh $TOPIC

