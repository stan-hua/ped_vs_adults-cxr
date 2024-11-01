#!/bin/bash -l
#SBATCH --job-name=ped_cxr_eval                   # Job name
#SBATCH --nodes=1                         # Number of nodes
#SBATCH --gres=gpu:NVIDIA_H100_80GB_HBM3:1                      # Request one GPU
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
#                                  Constants                                   #
################################################################################
# Specify experiment to evaluate
# EXP_NAME="exp_param_sweep-convnext_baseline-cardiomegaly-mixup-imb_sampler-bs=32"
EXP_NAME="exp_param_sweep-convnext_baseline-has_finding-mixup-imb_sampler-bs=32"

# Best/Last Checkpoint
CKPT_OPTION="best"


################################################################################
#                               Model Evaluation                               #
################################################################################
# Perform inference
# 1. VinDr-CXR (Val)
srun python -m src.scripts.eval_model \
    --task "infer" \
    --exp_name $EXP_NAME \
    --dset "vindr_cxr"\
    --split "val" \
    --ckpt_option $CKPT_OPTION \
    --use_comet_logger True

# 2. VinDr-CXR (Test)
srun python -m src.scripts.eval_model \
    --task "infer" \
    --exp_name $EXP_NAME \
    --dset "vindr_cxr"\
    --split "test" \
    --ckpt_option $CKPT_OPTION \
    --use_comet_logger True

# 3. VinDr-PCXR (Test)
srun python -m src.scripts.eval_model \
    --task "infer" \
    --exp_name $EXP_NAME \
    --dset "vindr_pcxr"\
    --split "test" \
    --ckpt_option $CKPT_OPTION \
    --use_comet_logger True

# 4. Check if child is over-predicted
srun python -m src.scripts.eval_model \
    --task "check_adult_vs_child" \
    --exp_name $EXP_NAME \
    --dset "vindr_pcxr"\
    --split "test" \
    --ckpt_option $CKPT_OPTION \
    --use_comet_logger True
