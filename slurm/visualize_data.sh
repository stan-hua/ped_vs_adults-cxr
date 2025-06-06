#!/bin/bash -l
#SBATCH --job-name=ped_cxr_viz            # Job name
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
#                                 Environment                                  #
################################################################################
# Load any necessary modules or activate your virtual environment here
micromamba activate peds_cxr


################################################################################
#                                Visualize Data                                #
################################################################################
srun python -m src.utils.data.viz_data age_histogram "vindr_pcxr"
# srun python -m src.utils.data.viz_data age_histogram "vindr_cxr" "padchest" "nih_cxr18" "chexbert"
# srun python -m src.utils.data.viz_data age_histogram "padchest" "nih_cxr18" --peds


srun python -m src.utils.data.viz_data pixel_histogram "vindr_pcxr"
# srun python -m src.utils.data.viz_data pixel_histogram "vindr_cxr" "padchest" "nih_cxr18" "chexbert"