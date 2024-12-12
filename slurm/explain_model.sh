#!/bin/bash -l
#SBATCH --job-name=ped_cxr_eval                   # Job name
#SBATCH --gres=gpu:1                      # Request one GPU
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
micromamba activate peds_cxr


################################################################################
#                                Explain Model                                 #
################################################################################
python -m src.utils.model.explain_model gradcam_cardiomegaly_on_vindr_pcxr