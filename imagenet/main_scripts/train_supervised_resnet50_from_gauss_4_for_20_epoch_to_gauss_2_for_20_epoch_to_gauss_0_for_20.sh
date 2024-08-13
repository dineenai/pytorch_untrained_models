#!/bin/bash
#
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=12
#SBATCH -J train_sup_RN50_from_gauss_4_for_20_epoch_to_gauss_2_for_20_epoch_to_gauss_0_for_20
#SBATCH --output=/data/blurry_vision_sup_RN50/supervised_resnet50_from_gauss_4_for_20_epoch_to_gauss_2_for_20_epoch_to_gauss_0_for_20/logs/slurm-%j.out
#SBATCH --error=/data/blurry_vision_sup_RN50/supervised_resnet50_from_gauss_4_for_20_epoch_to_gauss_2_for_20_epoch_to_gauss_0_for_20/logs/slurm-%j.err


DIR='/data/ILSVRC2012/'
OUTFOLDER="/data/blurry_vision_sup_RN50/supervised_resnet50_from_gauss_4_for_20_epoch_to_gauss_2_for_20_epoch_to_gauss_0_for_20/outmodel"
PYTHON="/opt/anaconda3/envs/blurry_vision/bin/python"
MODEL='resnet50'
EPOCHS=60
SAVE=5
RESUME="/data/blurry_vision_sup_RN50/supervised_resnet50_from_gauss_4_for_20_epoch_to_gauss_2_for_20_epoch_to_gauss_0_for_20/outmodel/checkpoint_supervised_resnet50_from_gauss_4_for_20_epoch_to_gauss_2_for_20_epoch_to_gauss_0_for_20_epoch40.pth.tar"
CPKT="supervised_resnet50_from_gauss_4_for_20_epoch_to_gauss_2_for_20_epoch_to_gauss_0_for_20"
GAUSS=0

${PYTHON}  main_general_copy.py --a ${MODEL} \
--model_path ${OUTFOLDER} \
--epochs ${EPOCHS} \
--save_freq ${SAVE} \
--resume ${RESUME} \
--cpkt_name ${CPKT} \
--gauss ${GAUSS} \
${DIR}