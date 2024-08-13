#!/bin/bash
#
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=12
#SBATCH -J model_accuracy
#SBATCH --output=/home/ainedineen/blurry_vision/pytorch_untrained_models/imagenet/model_accuracy/logs/slurm-%j.out
#SBATCH --error=/home/ainedineen/blurry_vision/pytorch_untrained_models/imagenet/model_accuracy/logs/slurm-%j.err


i=5

until [ $i -gt 60 ]
do

    echo i: $i
    
    DIR='/data/ILSVRC2012/'
    PYTHON="/opt/anaconda3/envs/blurry_vision/bin/python"
    PRINT=100
    BATCH=100
    MODEL='resnet50'
    RESUME="/data/blurry_vision_sup_RN50/supervised_resnet50_gauss_1_for_60_epoch/outmodel/checkpoint_supervised_resnet50_gauss_1_for_60_epoch_epoch${i}.pth.tar"
    AFILE="sup_RN50_gauss_1_for_60_epoch${i}"
    GAUSS=1

    ${PYTHON} main_val_model_accuracy_blur.py --a ${MODEL} \
    --print-freq ${PRINT} \
    --batch-size ${BATCH} \
    --resume ${RESUME} \
    --save_accuracy_file ${AFILE} \
    --gauss ${GAUSS} \
    --evaluate \
    ${DIR}
  
    ((i=i+5))
done

