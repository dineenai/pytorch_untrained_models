#!/bin/bash
#
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=12
#SBATCH -J model_accuracy
#SBATCH --output=/home/ainedineen/blurry_vision/pytorch_untrained_models/imagenet/model_accuracy/logs/slurm-%j.out
#SBATCH --error=/home/ainedineen/blurry_vision/pytorch_untrained_models/imagenet/model_accuracy/logs/slurm-%j.err

# starting epoch
i=1

# until [ $i -gt 60 ]
until [ $i -gt 90 ]
do

    echo i: $i
    
    DIR='/data/ILSVRC2012/'
    PYTHON="/opt/anaconda3/envs/blurry_vision/bin/python"
    PRINT=100
    BATCH=100
    MODEL='resnet50'
    # RESUME="/data/blurry_vision_sup_RN50/supervised_resnet50_from_gauss_0_for_30_epoch_to_gauss_4_for_30_epoch_to_gauss_0_for_30/outmodel/checkpoint_supervised_resnet50_from_gauss_0_for_30_epoch_to_gauss_4_for_30_epoch_to_gauss_0_for_30_epoch${i}.pth.tar"
    # AFILE="sup_RN50_from_gauss_0_for_30_epoch_to_gauss_4_for_30_epoch_to_gauss_0_for_30"
    # RESUME="/data/blurry_vision_sup_RN50/supervised_resnet50_from_gauss_4_for_30_epoch_to_gauss_0_for_30/outmodel/checkpoint_supervised_resnet50_from_gauss_4_for_30_epoch_to_gauss_0_for_30_epoch${i}.pth.tar"
    # AFILE="supervised_resnet50_from_gauss_4_for_30_epoch_to_gauss_0_for_30"
    
    # RESUME="/data/blurry_vision_sup_RN50/supervised_resnet50_gauss_0_for_5_epochs_changelr_1e/outmodel/checkpoint_supervised_resnet50_gauss_0_for_5_epochs_changelr_1e_epoch${i}.pth.tar"
    # AFILE="supervised_resnet50_from_gauss_0_for_5_epochs_changelr_1e"
    RESUME="/data/blurry_vision_sup_RN50/supervised_resnet50_gauss_0_1e/outmodel/checkpoint_supervised_resnet50_gauss_0_1e_epoch${i}.pth.tar"
    AFILE="supervised_resnet50_gauss_0_1e"
    # GAUSS=1

    ${PYTHON} main_val_model_accuracy_out_csv.py --a ${MODEL} \
    --print-freq ${PRINT} \
    --batch-size ${BATCH} \
    --resume ${RESUME} \
    --save_accuracy_file ${AFILE} \

    --evaluate \
    ${DIR}
  
    # ((i=i+5))
    ((i=i+1))
done

# /data/blurry_vision_sup_RN50/supervised_resnet50_gauss_0_for_5_epochs_changelr_1e/outmodel/checkpoint_supervised_resnet50_gauss_0_for_5_epochs_changelr_1e_epoch${i}.pth.tar