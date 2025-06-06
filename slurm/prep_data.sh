#!/bin/bash -l
#SBATCH --job-name=ped_cxr_prep_data                  # Job name
#SBATCH --nodes=1                         # Number of nodes
#SBATCH --cpus-per-task=8                 # Number of CPU cores per task
#SBATCH --mem=16GB
#SBATCH --tmp=16GB
#SBATCH -o slurm/logs/slurm-%j.out
#SBATCH --time=12:00:00

# If you want to do it in the terminal,
# salloc --job-name=my_shell --nodes=1 --cpus-per-task=6 --mem=16GB
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
micromamba activate peds_cxr


################################################################################
#                                Model Training                                #
################################################################################
srun python -m scripts.prep_data vindr_images --dset "vindr_cxr"
# srun python -m scripts.prep_data vindr_images --dset "vindr_pcxr"