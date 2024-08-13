#!/bin/bash
#
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=12
#SBATCH -J tiny_supervised_resnet50_tinyeyes_8week_29w_27d
#SBATCH --output=/data/blurry_vision_sup_RN50/supervised_resnet50_tinyeyes_8week_29w_27d/logs/slurm-%j.out
#SBATCH --error=/data/blurry_vision_sup_RN50/supervised_resnet50_tinyeyes_8week_29w_27d/logs/slurm-%j.err


DIR='/data/ILSVRC2012/'
OUTFOLDER="/data/blurry_vision_sup_RN50/supervised_resnet50_tinyeyes_8week_29w_27d/outmodel"
PYTHON="/opt/anaconda3/envs/blurry_vision/bin/python"
MODEL='resnet50'
EPOCHS=60
SAVE=1
CPKT="supervised_resnet50_tinyeyes_8week_29w_27d"
WEEK="week8"
WIDTH=27
DIST=29
RESUME="/data/blurry_vision_sup_RN50/supervised_resnet50_tinyeyes_8week_29w_27d/outmodel/checkpoint_supervised_resnet50_tinyeyes_8week_29w_27d_epoch20.pth.tar"

${PYTHON}  main_general_tinyeyes_oop.py --a ${MODEL} \
--model_path ${OUTFOLDER} \
--epochs ${EPOCHS} \
--save_freq ${SAVE} \
--cpkt_name ${CPKT} \
--tinyeyesweek ${WEEK} \
--tinyeyes_width ${WIDTH} \
--tinyeyes_dist ${DIST} \
--resume ${RESUME} \
${DIR}