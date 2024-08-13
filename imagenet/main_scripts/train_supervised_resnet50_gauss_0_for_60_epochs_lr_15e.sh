#!/bin/bash
#
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=12
#SBATCH -J blurry_supervised_resnet50_gauss_0_for_60_epoch_lr_15
#SBATCH --output=/data/blurry_vision_sup_RN50/supervised_resnet50_gauss_0_for_60_epoch_lr_15/logs/slurm-%j.out
#SBATCH --error=/data/blurry_vision_sup_RN50/supervised_resnet50_gauss_0_for_60_epoch_lr_15/logs/slurm-%j.err


DIR='/data/ILSVRC2012/'
OUTFOLDER="/data/blurry_vision_sup_RN50/supervised_resnet50_gauss_0_for_60_epoch_lr_15/outmodel"
PYTHON="/opt/anaconda3/envs/blurry_vision/bin/python"
MODEL='resnet50'
EPOCHS=60
SAVE=5
CPKT="supervised_resnet50_gauss_0_for_60_epoch_lr_15"
GAUSS=0

${PYTHON}  main_general_c_lr_15.py --a ${MODEL} \
--model_path ${OUTFOLDER} \
--epochs ${EPOCHS} \
--save_freq ${SAVE} \
--cpkt_name ${CPKT} \
--gauss ${GAUSS} \
${DIR}