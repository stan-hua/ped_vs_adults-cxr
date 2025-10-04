#!/bin/bash -l
#SBATCH --job-name=ped_cxr_eval                   # Job name
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
#                                  Constants                                   #
################################################################################
# Specify experiment to evaluate
EXP_NAMES=(
    "exp_cardiomegaly-vindr_cxr-mixup-imb_sampler"
    "exp_cardiomegaly-nih_cxr18-mixup-imb_sampler"
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

# Histogram matching blend ratio
# NOTE: Use 0 for no histogram matching
# TODO: Uncomment, if you'd like to re-perform inference with varying histogram matching
HM_RATIOS=(
    "0"
    # "0.25"
    # "0.5"
    # "0.75"
    # "1"
)


################################################################################
#                                  VinDr-CXR                                   #
################################################################################
# Perform inference
for EXP_NAME in "${EXP_NAMES[@]}"; do
    for HM_RATIO in "${HM_RATIOS[@]}"; do
        ############################################################################
        #                  Adult (Calibration & Healthy Set)                       #
        ############################################################################
        # 1. VinDr-CXR, NIH and PadChest
        for DSET in "${ADULT_DSETS[@]}"; do
            for SPLIT in "test_adult_calib" "test_healthy_adult"; do
                python -m scripts.eval_model main \
                    --task "infer" \
                    --exp_name $EXP_NAME \
                    --dset $DSET\
                    --split $SPLIT \
                    --transform_hm_blend_ratio $HM_RATIO \
                    --ckpt_option $CKPT_OPTION \
                    --use_comet_logger;
            done
        done

        ############################################################################
        #                               Children                                   #
        ############################################################################
        # 3. VinDr-PCXR
        python -m scripts.eval_model main \
            --task "infer" \
            --exp_name $EXP_NAME \
            --dset "vindr_pcxr"\
            --split "test" \
            --transform_hm_blend_ratio $HM_RATIO \
            --ckpt_option $CKPT_OPTION \
            --use_comet_logger;

        # 4. NIH and PadChest Children
        # for DSET in "${CHILDREN_DSETS[@]}"; do
        #     python -m scripts.eval_model main \
        #         --task "infer" \
        #         --exp_name $EXP_NAME \
        #         --dset $DSET \
        #         --split "test_peds" \
        #         --ckpt_option $CKPT_OPTION \
        #         --use_comet_logger;
        # done
    done
done
