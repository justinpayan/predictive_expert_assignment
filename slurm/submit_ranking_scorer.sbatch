#! /bin/bash
#SBATCH --job-name=specter_cs
#SBATCH --output=/work/pi_yzick_umass_edu/jpayan/logs/predictive_expert_assignment/specter_cs.out
#SBATCH --error=/work/pi_yzick_umass_edu/jpayan/logs/predictive_expert_assignment/specter_cs.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mem-per-gpu=32G
#SBATCH -G 1
#SBATCH --time=10-00:00:00
#SBATCH --partition=gpu-preempt

module load miniconda/22.11.1-1

conda activate llm_feat_annot

export CUR_PROJ_DIR=/work/pi_yzick_umass_edu/jpayan/predictive_expert_assignment

cd $CUR_PROJ_DIR/../docsim_reviewquals

./bin/pre_process/run_pp_gen_nearest.sh -a rank_pool -e specter -d cs -c