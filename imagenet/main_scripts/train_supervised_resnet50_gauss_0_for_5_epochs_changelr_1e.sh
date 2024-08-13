#!/bin/bash
#
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=12
#SBATCH -J blurry_supervised_resnet50_gauss_0_for_5_epochs_changelr_1e
#SBATCH --output=/data/blurry_vision_sup_RN50/supervised_resnet50_gauss_0_for_5_epochs_changelr_1e/logs/slurm-%j.out
#SBATCH --error=/data/blurry_vision_sup_RN50/supervised_resnet50_gauss_0_for_5_epochs_changelr_1e/logs/slurm-%j.err


DIR='/data/ILSVRC2012/'
OUTFOLDER="/data/blurry_vision_sup_RN50/supervised_resnet50_gauss_0_for_5_epochs_changelr_1e/outmodel"
PYTHON="/opt/anaconda3/envs/blurry_vision/bin/python"
MODEL='resnet50'
EPOCHS=5
CHANGELR=1
SAVE=1
CPKT="supervised_resnet50_gauss_0_for_5_epochs_changelr_1e"
GAUSS=0

${PYTHON}  main_general_c_lr_15.py --a ${MODEL} \
--model_path ${OUTFOLDER} \
--epochs ${EPOCHS} \
--save_freq ${SAVE} \
--change_lr ${CHANGELR} \
--cpkt_name ${CPKT} \
--gauss ${GAUSS} \
${DIR}


