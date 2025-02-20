#!/bin/bash
#
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=12
#SBATCH -J mid_train_log
#SBATCH --output=/data2/blurry_vision_sup_RN50/supervised_alexnet_bp_butter_mid_batches_log_for_60_epoch_params-AN_default_bs-32_res-fixed_iter-1/logs/slurm-%j.out
#SBATCH --error=/data2/blurry_vision_sup_RN50/supervised_alexnet_bp_butter_mid_batches_log_for_60_epoch_params-AN_default_bs-32_res-fixed_iter-1/logs/slurm-%j.err

# Added res-fixed 21-11-24 
PARAMDESCRIP="params-AN_default_bs-32_res-fixed"
ITER=1
BAND='mid'
OUTPTH="/data2/blurry_vision_sup_RN50/supervised_alexnet_bp_butter_${BAND}_batches_log_for_60_epoch_${PARAMDESCRIP}_iter-${ITER}"



DIR="/data2/ILSVRC2012/butterworth/cut-0.055-0.15_order-3.0_npad-40/${BAND}"
# DIR="/data/ILSVRC2012" # Fixed 21-11-24 : added res-fixed above
OUTFOLDER="${OUTPTH}/outmodel"
PYTHON="/opt/anaconda3/envs/blurry_vision/bin/python"
MODEL='alexnet'
# CPKT="supervised_alexnet_bp_butter_${BAND}_batches_for_60_epoch_lr-${LRFREQ}"
CPKT="supervised_alexnet_bp_butter_${BAND}_batches_for_60_epoch_${PARAMDESCRIP}"
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