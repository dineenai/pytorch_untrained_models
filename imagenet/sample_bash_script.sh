#!/bin/bash
#
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=12
#SBATCH -J training_supervised_resnet50_on_ImageNet
#SBATCH --output=/data/aines_custom_trained_models/supervised_resnet50_trained_on_ImageNet/logs/slurm-%j.out
#SBATCH --error=/data/aines_custom_trained_models/supervised_resnet50_trained_on_ImageNet/logs/slurm-%j.err

DIR='/data2/ILSVRC2012'
OUTFOLDER="/data/aines_custom_trained_models/supervised_resnet50_trained_on_ImageNet/outmodel"
PYTHON="/opt/anaconda3/envs/blurry_vision/bin/python"
MODEL='resnet50'
EPOCHS=60
CPKT="supervised_resnet50_trained-for_${EPOCHS}_epochs"


${PYTHON} main.py --a ${MODEL} \
--model_path ${OUTFOLDER} \
--epochs ${EPOCHS} \
--cpkt_name ${CPKT} \
${DIR}



