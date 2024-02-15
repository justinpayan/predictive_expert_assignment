#! /bin/bash
#SBATCH --job-name=llm_feat_annot
#SBATCH --output=/work/pi_yzick_umass_edu/jpayan/logs/predictive_expert_assignment/feat_annot_cs.out
#SBATCH --error=/work/pi_yzick_umass_edu/jpayan/logs/predictive_expert_assignment/feat_annot_cs.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mem-per-gpu=32G
#SBATCH -G 1
#SBATCH --time=10-00:00:00
#SBATCH --partition=gpu-preempt
#SBATCH --constraint=rtx8000|v100|a100

./submit_feat_annot.sh
