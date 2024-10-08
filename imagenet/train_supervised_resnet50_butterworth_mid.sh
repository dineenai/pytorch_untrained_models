#!/bin/bash
#
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=12
#SBATCH -J blurry_train_supervised_resnet50_bp_butter_mid_for_60
#SBATCH --output=/data/blurry_vision_sup_RN50/supervised_resnet50_bp_butter_mid_for_60_epoch/logs/slurm-%j.out
#SBATCH --error=/data/blurry_vision_sup_RN50/supervised_resnet50_bp_butter_mid_for_60_epoch/logs/slurm-%j.err


DIR='/data2/ILSVRC2012/butterworth/cut-0.055-0.15_order-3.0_npad-40/mid'
OUTFOLDER="/data/blurry_vision_sup_RN50/supervised_resnet50_bp_butter_mid_for_60_epoch/outmodel"
PYTHON="/opt/anaconda3/envs/blurry_vision/bin/python"
MODEL='resnet50'
CPKT="supervised_resnet50_bp_butter_mid_for_60_epoch"
EPOCHS=30
SAVE=1



${PYTHON} main_general_copy_mmk_pared_back.py --a ${MODEL} \
--model_path ${OUTFOLDER} \
--epochs ${EPOCHS} \
--cpkt_name ${CPKT} \
--save_freq ${SAVE} \
${DIR}