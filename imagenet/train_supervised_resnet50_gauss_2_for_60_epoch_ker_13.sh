#!/bin/bash
#
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=12
#SBATCH -J blurry_train_supervised_resnet50_gauss_2_for_60_epoch_ker_13
#SBATCH --output=/data/blurry_vision_sup_RN50/supervised_resnet50_gauss_2_for_60_epoch_ker_13/logs/slurm-%j.out
#SBATCH --error=/data/blurry_vision_sup_RN50/supervised_resnet50_gauss_2_for_60_epoch_ker_13/logs/slurm-%j.err


DIR='/data/ILSVRC2012/'
OUTFOLDER="/data/blurry_vision_sup_RN50/supervised_resnet50_gauss_2_for_60_epoch_ker_13/outmodel"
PYTHON="/opt/anaconda3/envs/blurry_vision/bin/python"
MODEL='resnet50'
CPKT="supervised_resnet50_gauss_2_for_60_epoch_ker_13"
EPOCHS=60
SAVE=5
GAUSS=2
KER=13
MMK=0

${PYTHON} main_general_copy_mmk.py --a ${MODEL} \
--model_path ${OUTFOLDER} \
--epochs ${EPOCHS} \
--gauss ${GAUSS} \
--kernel ${KER} \
--cpkt_name ${CPKT} \
--save_freq ${SAVE} \
${DIR}