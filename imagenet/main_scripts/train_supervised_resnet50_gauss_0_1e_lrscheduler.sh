#!/bin/bash
#
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=12
#SBATCH -J blurry_supervised_resnet50_gauss_0_1e_cosineannealinglr
#SBATCH --output=/data/blurry_vision_sup_RN50/supervised_resnet50_gauss_0_1e_cosineannealinglr/logs/slurm-%j.out
#SBATCH --error=/data/blurry_vision_sup_RN50/supervised_resnet50_gauss_0_1e_cosineannealinglr/logs/slurm-%j.err


DIR='/data/ILSVRC2012/'
OUTFOLDER="/data/blurry_vision_sup_RN50/supervised_resnet50_gauss_0_1e_cosineannealinglr/outmodel"
PYTHON="/opt/anaconda3/envs/aine_pytorch_oct23/bin/python"
MODEL='resnet50'
EPOCHS=90
SAVE=1
CPKT="supervised_resnet50_gauss_0_1e_cosineannealinglr"
GAUSS=0
LRSCHEDULER="cosineannealinglr"
LRWARMUPEPOCHS=5
LRWARMUPMETHOD="linear"



${PYTHON}  main_general_copy_lrscheduler.py --a ${MODEL} \
--model_path ${OUTFOLDER} \
--epochs ${EPOCHS} \
--save_freq ${SAVE} \
--cpkt_name ${CPKT} \
--gauss ${GAUSS} \
--lr-scheduler ${LRSCHEDULER} \
--lr-warmup-epochs ${LRWARMUPEPOCHS} \
--lr-warmup-method ${LRWARMUPMETHOD} \
${DIR}

