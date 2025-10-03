#!/bin/bash -l
#SBATCH --job-name=ped_cxr_train                  # Job name
#SBATCH --nodes=1                         # Number of nodes
#SBATCH --gres=gpu:1          # Request one GPU
#SBATCH --cpus-per-task=6                 # Number of CPU cores per task
#SBATCH --mem=16GB
#SBATCH --tmp=8GB
#SBATCH -o slurm/logs/slurm-%j.out
#SBATCH --time=12:00:00

# If you want to do it in the terminal,
# salloc --job-name=my_shell --nodes=1 --gres=gpu:NVIDIA_L40S:1 --cpus-per-task=6 --mem=32GB
# srun (command)

################################################################################
#                                   Re-Queue                                   #
################################################################################
# send this batch script a SIGUSR1 60 seconds
# before we hit our time limit
#SBATCH --signal=B:USR1@60
# trap handler - resubmit ourselves

# Trap handler for SIGUSR1 which is sent 60 seconds before the job times out.
# It requeues itself so that it can continue running.
handler(){
    echo "function handler called at $(date)"
    # do whatever cleanup you want here;
    # checkpoint, sync, etc
    # sbatch "$0"
    scontrol requeue $SLURM_JOB_ID
}
# register signal handler
trap 'handler' SIGUSR1


################################################################################
#                                 Environment                                  #
################################################################################
# Load any necessary modules or activate your virtual environment here

# Option 1. pixi
if [[ "$USE_PIXI" -eq 1 ]]; then
    pixi shell -e torch-gpu
# Option 2. Conda
else
    conda activate peds_cxr
fi

################################################################################
#                                Model Training                                #
################################################################################
python -m scripts.train_model -c "exp_cardiomegaly-vindr_cxr-mixup-imb_sampler.ini"
# python -m scripts.train_model -c "exp_cardiomegaly-nih_cxr18-mixup-imb_sampler.ini"
# python -m scripts.train_model -c "exp_cardiomegaly-padchest-mixup-imb_sampler.ini"
# python -m scripts.train_model -c "exp_cardiomegaly-chexbert-mixup-imb_sampler.ini"