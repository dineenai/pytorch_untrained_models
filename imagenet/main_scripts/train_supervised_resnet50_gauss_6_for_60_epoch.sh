#!/bin/bash
#
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=12
#SBATCH -J blurry_train_supervised_resnet50_gauss_6_for_60_epoch
#SBATCH --output=/data/blurry_vision_sup_RN50/supervised_resnet50_gauss_6_for_60_epoch/logs/slurm-%j.out
#SBATCH --error=/data/blurry_vision_sup_RN50/supervised_resnet50_gauss_6_for_60_epoch/logs/slurm-%j.err


DIR='/data/ILSVRC2012/'
OUTFOLDER="/data/blurry_vision_sup_RN50/supervised_resnet50_gauss_6_for_60_epoch/outmodel"
PYTHON="/opt/anaconda3/envs/blurry_vision/bin/python"
MODEL='resnet50'
EPOCHS=60
SAVE=5
NAME='supervised_resnet50_gauss_6_for_60_epoch'
RESUME='/data/blurry_vision_sup_RN50/supervised_resnet50_gauss_6_for_60_epoch/outmodel/checkpoint_supervised_resnet50_gauss_6_for_60_epoch_epoch50.pth.tar'

${PYTHON} main_general.py --a ${MODEL} \
--model_path ${OUTFOLDER} \
--epochs ${EPOCHS} \
--save_freq ${SAVE} \
--cpkt_name ${NAME} \
--resume ${RESUME} \
${DIR}