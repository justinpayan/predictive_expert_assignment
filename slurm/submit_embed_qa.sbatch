#! /bin/bash
#SBATCH --job-name=embed_qa
#SBATCH --output=/work/pi_yzick_umass_edu/jpayan/logs/predictive_expert_assignment/embed_qa_cs.out
#SBATCH --error=/work/pi_yzick_umass_edu/jpayan/logs/predictive_expert_assignment/embed_qa_cs.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mem-per-gpu=32G
#SBATCH -G 1
#SBATCH --time=3-00:00:00
#SBATCH --partition=gpu-preempt
#SBATCH --constraint=rtx8000|v100|a100

./submit_embed_qa.sh