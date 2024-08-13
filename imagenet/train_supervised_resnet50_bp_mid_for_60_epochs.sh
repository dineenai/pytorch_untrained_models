#!/bin/bash
#
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=12
#SBATCH -J blurry_train_supervised_resnet50_bp_mid_for_60_epoch
#SBATCH --output=/data/blurry_vision_sup_RN50/supervised_resnet50_bp_mid_for_60_epoch/logs/slurm-%j.out
#SBATCH --error=/data/blurry_vision_sup_RN50/supervised_resnet50_bp_mid_for_60_epoch/logs/slurm-%j.err


DIR='/data2/ILSVRC2012/bandpass/mid'
OUTFOLDER="/data/blurry_vision_sup_RN50/supervised_resnet50_bp_mid_for_60_epoch/outmodel"
PYTHON="/opt/anaconda3/envs/blurry_vision/bin/python"
MODEL='resnet50'
EPOCHS=60
CPKT="supervised_resnet50_bp_mid_for_60_epoch"
SAVE=5
RESUME="/data/blurry_vision_sup_RN50/supervised_resnet50_bp_mid_for_60_epoch/outmodel/checkpoint_supervised_resnet50_bp_mid_for_60_epoch_epoch25.pth.tar"



${PYTHON} main_general_copy_bp.py --a ${MODEL} \
--model_path ${OUTFOLDER} \
--epochs ${EPOCHS} \
--save_freq ${SAVE} \
--cpkt_name ${CPKT} \
--resume ${RESUME} \
${DIR}