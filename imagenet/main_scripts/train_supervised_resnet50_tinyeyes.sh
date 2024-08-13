#!/bin/bash
#
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=12
#SBATCH -J tiny_supervised_resnet50_tinyeyes_8week_15.8w_60d
#SBATCH --output=/data/blurry_vision_sup_RN50/ssupervised_resnet50_tinyeyes_8week_15.8w_60d/logs/slurm-%j.out
#SBATCH --error=/data/blurry_vision_sup_RN50/supervised_resnet50_tinyeyes_8week_15.8w_60d/logs/slurm-%j.err


DIR='/data/ILSVRC2012/'
OUTFOLDER="/data/blurry_vision_sup_RN50/supervised_resnet50_tinyeyes_8week_15.8w_60d/outmodel"
PYTHON="/opt/anaconda3/envs/blurry_vision/bin/python"
MODEL='resnet50'
EPOCHS=10
SAVE=1
CPKT="supervised_resnet50_tinyeyes_8week_15.8w_60d"
WEEK="week8"
WIDTH=15.8
RESUME="/data/blurry_vision_sup_RN50/supervised_resnet50_tinyeyes_8week_15.8w_60d/outmodel/checkpoint_supervised_resnet50_tinyeyes_8week_15.8w_60d_epoch1.pth.tar"

${PYTHON}  main_general_tinyeyes.py --a ${MODEL} \
--model_path ${OUTFOLDER} \
--epochs ${EPOCHS} \
--save_freq ${SAVE} \
--resume ${RESUME} \
--cpkt_name ${CPKT} \
--tinyeyesweek ${WEEK} \
--tinyeyes_width ${WIDTH} \
${DIR}