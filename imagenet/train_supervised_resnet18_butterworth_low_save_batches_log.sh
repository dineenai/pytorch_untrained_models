#!/bin/bash
#
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=12
#SBATCH -J all_train_log
#SBATCH --output=/data2/blurry_vision_sup_RN50/supervised_resnet18_bp_butter_low_batches_log_for_60_epoch_iter-1/logs/slurm-%j.out
#SBATCH --error=/data2/blurry_vision_sup_RN50/supervised_resnet18_bp_butter_low_batches_log_for_60_epoch_iter-1/logs/slurm-%j.err



# ITER=5
ITER=1
BAND='low'
OUTPTH="/data2/blurry_vision_sup_RN50/supervised_resnet18_bp_butter_${BAND}_batches_log_for_60_epoch_iter-${ITER}"

# mkdir -p ${OUTPTH}/logs
# mkdir ${OUTPTH}/outmodel


DIR="/data2/ILSVRC2012/butterworth/cut-0.055-0.15_order-3.0_npad-40/${BAND}"
OUTFOLDER="${OUTPTH}/outmodel"
PYTHON="/opt/anaconda3/envs/blurry_vision/bin/python"
MODEL='resnet18'
CPKT="supervised_resnet18_bp_butter_${BAND}_batches_for_60_epoch"
EPOCHS=90
SAVE=1
ACCPATH="/home/ainedineen/blurry_vision/pytorch_untrained_models/imagenet/bandpass_analysis_butterworth/batched_accuracy"
PRINTFREQ=100
BATCHNOPATH="/home/ainedineen/blurry_vision/pytorch_untrained_models/imagenet/log_batches.csv"

${PYTHON} main_general_copy_mmk_pared_back_save_in_epoch_save_log_batches.py --a ${MODEL} \
--model_path ${OUTFOLDER} \
--epochs ${EPOCHS} \
--cpkt_name ${CPKT} \
--save_freq ${SAVE} \
--print-freq ${PRINTFREQ} \
--path_acc ${ACCPATH} \
--iter ${ITER} \
--batches_to_save ${BATCHNOPATH} \
${DIR}