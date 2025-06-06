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
micromamba activate peds_cxr


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
                srun python -m scripts.eval_model main \
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
        srun python -m scripts.eval_model main \
            --task "infer" \
            --exp_name $EXP_NAME \
            --dset "vindr_pcxr"\
            --split "test" \
            --transform_hm_blend_ratio $HM_RATIO \
            --ckpt_option $CKPT_OPTION \
            --use_comet_logger;

        # 4. NIH and PadChest Children
        # for DSET in "${CHILDREN_DSETS[@]}"; do
        #     srun python -m scripts.eval_model main \
        #         --task "infer" \
        #         --exp_name $EXP_NAME \
        #         --dset $DSET \
        #         --split "test_peds" \
        #         --ckpt_option $CKPT_OPTION \
        #         --use_comet_logger;
        # done
    done
done

# 4. Check if child is over-predicted
HM_RATIO=0
for EXP_NAME in "${EXP_NAMES[@]}"; do
    python -m scripts.eval_model main \
        --task "check_adult_vs_child" \
        --exp_name $EXP_NAME \
        --dset "vindr_pcxr"\
        --split "test" \
        --transform_hm_blend_ratio $HM_RATIO \
        --ckpt_option $CKPT_OPTION \
        --use_comet_logger
done

# 4.2. Check if child is over-predicted (aggregated)
python -m scripts.eval_model check_child_fpr "${EXP_NAMES[@]}" --transform_hm_blend_ratio $HM_RATIO

# 5. Check if adults are over-predicted
python -m scripts.eval_model check_adult_fpr_same "${EXP_NAMES[@]}" --transform_hm_blend_ratio $HM_RATIO
python -m scripts.eval_model check_adult_fpr_diff "${EXP_NAMES[@]}" --transform_hm_blend_ratio $HM_RATIO

# 6. Create Figure 2: Cardiomegaly False Positive Rates by Page
python -m scripts.eval_model check_child_and_adult_fpr

# 7. Create Supp. Figure: Check impact of histogram matching
python -m scripts.eval_model impact_of_hm "${EXP_NAMES[@]}"
