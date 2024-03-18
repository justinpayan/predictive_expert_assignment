#! /bin/bash
#SBATCH --job-name=ex_shap
#SBATCH --output=/mnt/nfs/scratch1/jpayan/logs/predictive_expert_assignment/ex_shap.out
#SBATCH --error=/mnt/nfs/scratch1/jpayan/logs/predictive_expert_assignment/ex_shap.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --time=6-00:00:00
#SBATCH --partition=longq

./submit_exhaustive_shapley.sh