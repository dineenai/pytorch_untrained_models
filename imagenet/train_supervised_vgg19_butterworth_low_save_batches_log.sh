#!/bin/bash
#
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=12
#SBATCH -J low_train_log
#SBATCH --output=/data2/blurry_vision_sup_RN50/supervised_vgg19_bp_butter_low_batches_log_for_60_epoch_params-AN_default_bs-32_iter-1/logs/slurm-%j.out
#SBATCH --error=/data2/blurry_vision_sup_RN50/supervised_vgg19_bp_butter_low_batches_log_for_60_epoch_params-AN_default_bs-32_iter-1/logs/slurm-%j.err

PARAMDESCRIP="params-AN_default_bs-32"
ITER=1
BAND='low'
OUTPTH="/data2/blurry_vision_sup_RN50/supervised_vgg19_bp_butter_${BAND}_batches_log_for_60_epoch_${PARAMDESCRIP}_iter-${ITER}"



DIR="/data/ILSVRC2012"
OUTFOLDER="${OUTPTH}/outmodel"
PYTHON="/opt/anaconda3/envs/blurry_vision/bin/python"
MODEL='vgg19'
# CPKT="supervised_vgg19_bp_butter_${BAND}_batches_for_60_epoch_lr-${LRFREQ}"
CPKT="supervised_vgg19_bp_butter_${BAND}_batches_for_60_epoch_${PARAMDESCRIP}"
EPOCHS=90
SAVE=1
ACCPATH="/home/ainedineen/blurry_vision/pytorch_untrained_models/imagenet/bandpass_analysis_butterworth/batched_accuracy"
PRINTFREQ=100
BATCHNOPATH="/home/ainedineen/blurry_vision/pytorch_untrained_models/imagenet/log_batches.csv"
LR=0.01
BS=32

${PYTHON} main_general_copy_mmk_pared_back_save_in_epoch_save_log_batches.py --a ${MODEL} \
--model_path ${OUTFOLDER} \
--epochs ${EPOCHS} \
--lr ${LR} \
--cpkt_name ${CPKT} \
--save_freq ${SAVE} \
--print-freq ${PRINTFREQ} \
--path_acc ${ACCPATH} \
--iter ${ITER} \
--batch-size ${BS} \
--batches_to_save ${BATCHNOPATH} \
${DIR}