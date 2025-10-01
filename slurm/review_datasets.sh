#!/bin/bash -l
#SBATCH --job-name=dataset_eval                   # Job name
# --gres=gpu:1                      # Request one GPU
#SBATCH --nodes=1                         # Number of nodes
#SBATCH --cpus-per-task=6                 # Number of CPU cores per task
#SBATCH --mem=8GB
#SBATCH -o slurm/logs/slurm-%j.out

# If you want to do it in the terminal,
# salloc --job-name=my_shell --nodes=1 --gres=gpu:1 --cpus-per-task=6 --mem=32GB
# srun (command)

################################################################################
#                              Setup Environment                               #
################################################################################
# Option 1. pixi
if [[ "$USE_PIXI" -eq 1 ]]; then
    pixi shell -e torch-gpu
# Option 2. Conda
else
    conda activate peds_cxr
fi


################################################################################
#                               Perform Analysis                               #
################################################################################
python -m scripts.describe_data describe_datasets
# python -m scripts.describe_data describe_peds_in_each_category
# python -m scripts.describe_data describe_collections
# python -m scripts.describe_data describe_peds_broken_by_modality_task
# python -m scripts.describe_data describe_peds_broken_by_modality_task --peds_vs_adult False
# python -m scripts.describe_data describe_peds_broken_by_modality_task --modality "XR"
# python -m scripts.describe_data describe_peds_broken_by
