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
#                                  Constants                                   #
################################################################################
# Specify experiment to evaluate
EXP_NAMES=(
    "exp_cardiomegaly-vindr_cxr-mixup-imb_sampler"
    "exp_cardiomegaly-nih_cxr-mixup-imb_sampler"
    "exp_cardiomegaly-padchest-mixup-imb_sampler"
    "exp_cardiomegaly-chexbert-mixup-imb_sampler"
)

# Best/Last Checkpoint
CKPT_OPTION="best"

ADULT_DSETS=(
    "vindr_cxr"
    "nih_cxr18"
    "padchest"
    "chexbert"
)
# NOTE: VinDr-PCXR is always run
CHILDREN_DSETS=(
    "nih_cxr18"
    "padchest"
)


################################################################################
#                                  VinDr-CXR                                   #
################################################################################
# Perform inference
for EXP_NAME in "${EXP_NAMES[@]}"; do
    ############################################################################
    #                  Adult (Calibration & Healthy Set)                       #
    ############################################################################
    # 1. VinDr-CXR, NIH and PadChest
    # for DSET in "${ADULT_DSETS[@]}"; do
    #     for SPLIT in "test_adult_calib" "test_healthy_adult"; do
    #         srun python -m src.scripts.eval_model main \
    #             --task "infer" \
    #             --exp_name $EXP_NAME \
    #             --dset $DSET\
    #             --split $SPLIT \
    #             --ckpt_option $CKPT_OPTION \
    #             --use_comet_logger;
    #     done
    # done

    ############################################################################
    #                               Children                                   #
    ############################################################################
    # 3. VinDr-PCXR
    srun python -m src.scripts.eval_model main \
        --task "infer" \
        --exp_name $EXP_NAME \
        --dset "vindr_pcxr"\
        --split "test" \
        --ckpt_option $CKPT_OPTION \
        --use_comet_logger;

    # 4. NIH and PadChest
    # for DSET in "${CHILDREN_DSETS[@]}"; do
    #     srun python -m src.scripts.eval_model main \
    #         --task "infer" \
    #         --exp_name $EXP_NAME \
    #         --dset $DSET \
    #         --split "test_peds" \
    #         --ckpt_option $CKPT_OPTION \
    #         --use_comet_logger;
    # done
done

# 4. Check if child is over-predicted
for EXP_NAME in "${EXP_NAMES[@]}"; do
    python -m src.scripts.eval_model main \
        --task "check_adult_vs_child" \
        --exp_name $EXP_NAME \
        --dset "vindr_pcxr"\
        --split "test" \
        --ckpt_option $CKPT_OPTION \
        --use_comet_logger
done

# 5. Check if adults are over-predicted
python -m src.scripts.eval_model check_adult_fpr "${EXP_NAMES[@]}"
