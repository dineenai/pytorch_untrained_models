#!/bin/bash
#
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=6
#SBATCH -J blurry_supervised_resnet50_from_gauss_4_for_30_epoch_to_gauss_0_for_30_TRAIN2
#SBATCH --output=/data/blurry_vision_sup_RN50/supervised_resnet50_from_gauss_4_for_30_epoch_to_gauss_0_for_30_TRAIN2/logs/slurm-%j.out
#SBATCH --error=/data/blurry_vision_sup_RN50/supervised_resnet50_from_gauss_4_for_30_epoch_to_gauss_0_for_30_TRAIN2/logs/slurm-%j.err


DIR='/data/ILSVRC2012/'
OUTFOLDER="/data/blurry_vision_sup_RN50/supervised_resnet50_from_gauss_4_for_30_epoch_to_gauss_0_for_30_TRAIN2/outmodel"
PYTHON="/opt/anaconda3/envs/blurry_vision/bin/python"
MODEL='resnet50'
EPOCHS=60
SAVE=5
CPKT="supervised_resnet50_from_gauss_4_for_30_epoch_to_gauss_0_for_30_TRAIN2"
RESUME="/data/blurry_vision_sup_RN50/supervised_resnet50_from_gauss_4_for_30_epoch_to_gauss_0_for_30_TRAIN2/outmodel/checkpoint_supervised_resnet50_from_gauss_4_for_30_epoch_to_gauss_0_for_30_TRAIN2_epoch30.pth.tar"
GAUSS=0


${PYTHON}  main_general_copy.py --a ${MODEL} \
--model_path ${OUTFOLDER} \
--epochs ${EPOCHS} \
--save_freq ${SAVE} \
--cpkt_name ${CPKT} \
--gauss ${GAUSS} \
--resume ${RESUME} \
${DIR}