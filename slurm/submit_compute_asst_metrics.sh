#!/usr/bin/env bash

module load gurobi/1001

SEED=$1

python ../src/compute_asst_metrics.py --topic cs --seed $SEED