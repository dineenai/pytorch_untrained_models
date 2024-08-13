#!/bin/bash
#
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=12
#SBATCH -J blurry_supervised_resnet50_from_gauss_2_for_30_epoch_to_gauss_0_for_30
#SBATCH --output=/home/ainedineen/blurry_vision/pytorch_untrained_models/imagenet/supervised_resnet50_from_gauss_2_for_30_epoch_to_gauss_0_for_30/logs/slurm-%j.out
#SBATCH --error=/home/ainedineen/blurry_vision/pytorch_untrained_models/imagenet/supervised_resnet50_from_gauss_2_for_30_epoch_to_gauss_0_for_30/logs/slurm-%j.err


DIR='/data/ILSVRC2012/'
OUTFOLDER="/home/ainedineen/blurry_vision/pytorch_untrained_models/imagenet/supervised_resnet50_from_gauss_2_for_30_epoch_to_gauss_0_for_30/outmodel"
PYTHON="/opt/anaconda3/envs/blurry_vision/bin/python"
MODEL='resnet50'
EPOCHS=60
SAVE=5
RESUME="/home/ainedineen/blurry_vision/pytorch_untrained_models/imagenet/supervised_resnet50_gauss_2_for_60_epoch/outmodel/checkpoint_supervised_resnet50_gauss_2_for_60_epoch30.pth.tar"

${PYTHON}  main_supervised_resnet50_from_gauss_2_for_30_epoch_to_gauss_0_for_30_pt2.py --a ${MODEL} \
--model_path ${OUTFOLDER} \
--epochs ${EPOCHS} \
--save_freq ${SAVE} \
--resume ${RESUME} \
${DIR}