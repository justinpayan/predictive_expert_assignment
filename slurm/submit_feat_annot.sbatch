#! /bin/bash

TOPIC=$1

sbatch \
--job-name=${TOPIC}_llm_feat_annot \
--output=/work/pi_yzick_umass_edu/jpayan/logs/predictive_expert_assignment/feat_annot_${TOPIC}.out \
--error=/work/pi_yzick_umass_edu/jpayan/logs/predictive_expert_assignment/feat_annot_${TOPIC}.err \
--nodes=1 \
--ntasks=1 \
--mem-per-gpu=32G \
-G 1 \
--time=10-00:00:00 \
--partition=gpu-preempt \
--constraint="rtx8000|a100" \
./submit_feat_annot.sh $TOPIC
