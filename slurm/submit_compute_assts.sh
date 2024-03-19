#!/usr/bin/env bash

module load gurobi/1001

SEED=$1

python ../src/compute_assts.py --topic cs --seed $SEED



