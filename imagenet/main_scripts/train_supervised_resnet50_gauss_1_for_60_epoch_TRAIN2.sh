#!/bin/bash
#
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=12
#SBATCH -J blurry_supervised_resnet50_gauss_1_for_60_epoch
#SBATCH --output=/data/blurry_vision_sup_RN50/supervised_resnet50_gauss_1_for_60_epoch_TRAIN2/logs/slurm-%j.out
#SBATCH --error=/data/blurry_vision_sup_RN50/supervised_resnet50_gauss_1_for_60_epoch_TRAIN2/logs/slurm-%j.err


DIR='/data/ILSVRC2012/'
OUTFOLDER="/data/blurry_vision_sup_RN50/supervised_resnet50_gauss_1_for_60_epoch_TRAIN2/outmodel"
PYTHON="/opt/anaconda3/envs/blurry_vision/bin/python"
MODEL='resnet50'
EPOCHS=60
SAVE=1
RESUME="/data/blurry_vision_sup_RN50/supervised_resnet50_gauss_1_for_60_epoch_TRAIN2/outmodel/checkpoint_supervised_resnet50_gauss_1_for_60_epoch_TRAIN2_epoch15.pth.tar"
CPKT="supervised_resnet50_gauss_1_for_60_epoch_TRAIN2"
GAUSS=1

${PYTHON}  main_general_copy.py --a ${MODEL} \
--model_path ${OUTFOLDER} \
--epochs ${EPOCHS} \
--save_freq ${SAVE} \
--resume ${RESUME} \
--cpkt_name ${CPKT} \
--gauss ${GAUSS} \
${DIR}