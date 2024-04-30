#!/usr/bin/env bash

module load gurobi/1001

TOPIC=$1
SEED=$2

python ../src/compute_asst_metrics.py --topic $TOPIC --seed $SEED