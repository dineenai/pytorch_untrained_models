#!/bin/bash
#
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=12
#SBATCH -J blurry_val_supervised_ResNet50_elephantcat
#SBATCH --output=/home/ainedineen/blurry_vision/pytorch_untrained_models/imagenet/supervised_resnet50/evaluate/logs/slurm-%j.out
#SBATCH --error=/home/ainedineen/blurry_vision/pytorch_untrained_models/imagenet/supervised_resnet50/evaluate/logs/slurm-%j.err


DIR='/data/ILSVRC2012/'
PYTHON="/opt/anaconda3/envs/blurry_vision/bin/python"
PRINT=100
BATCH=100
MODEL='resnet50'
RESUME="/data/blurry_vision_sup_RN50/supervised_resnet50_gauss_6_for_60_epoch/outmodel/checkpoint_supervised_resnet50_gauss_6_for_60_epoch_epoch30.pth.tar"
AFILE="gauss_6_val_sup_RN50_gauss_6_for_60_epoch_epoch30"
GAUSS=6

${PYTHON} main_val_model_accuracy_blur.py --a ${MODEL} \
--print-freq ${PRINT} \
--batch-size ${BATCH} \
--resume ${RESUME} \
--save_accuracy_file ${AFILE} \
--gauss ${GAUSS} \
--evaluate \
${DIR}

