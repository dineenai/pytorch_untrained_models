#!/bin/bash
#
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=12
#SBATCH -J blurry_train_supervised_resnet50_bp_butter_low_batches_for_60
#SBATCH --output=/data/blurry_vision_sup_RN50/supervised_resnet50_bp_butter_low_batches_for_60_epoch/logs/slurm-%j.out
#SBATCH --error=/data/blurry_vision_sup_RN50/supervised_resnet50_bp_butter_low_batches_for_60_epoch/logs/slurm-%j.err


DIR='/data2/ILSVRC2012/butterworth/cut-0.055-0.15_order-3.0_npad-40/low'
OUTFOLDER="/data/blurry_vision_sup_RN50/supervised_resnet50_bp_butter_low_batches_for_60_epoch/outmodel"
PYTHON="/opt/anaconda3/envs/blurry_vision/bin/python"
MODEL='resnet50'
CPKT="supervised_resnet50_bp_butter_low_batches_for_60_epoch"
EPOCHS=15
SAVE=1
ACCPATH="/home/ainedineen/blurry_vision/pytorch_untrained_models/imagenet/bandpass_analysis_butterworth/batched_accuracy"
PRINTFREQ=100

${PYTHON} main_general_copy_mmk_pared_back_save_in_epoch.py --a ${MODEL} \
--model_path ${OUTFOLDER} \
--epochs ${EPOCHS} \
--cpkt_name ${CPKT} \
--save_freq ${SAVE} \
--print-freq ${PRINTFREQ} \
--path_acc ${ACCPATH} \
${DIR}