#!/bin/bash
#
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=12
#SBATCH -J shape_bias
#SBATCH --output=/home/ainedineen/blurry_vision/pytorch_untrained_models/imagenet/shape_bias/logs/slurm-%j.out
#SBATCH --error=/home/ainedineen/blurry_vision/pytorch_untrained_models/imagenet/shape_bias/logs/slurm-%j.err

i=5


until [ $i -gt 60 ]
do

    echo i: $i
    echo j: $j

    DIR='/home/ainedineen/blurry_vision/texture-vs-shape/stimuli/'
    PYTHON="/opt/anaconda3/envs/blurry_vision/bin/python"
    PRINT=1
    BATCH=1
    MODEL='resnet50'

    # AFILE="sup_RN50_gauss_1_for_60_epoch${i}"
    # AFILE="sup_RN50_gauss_0_for_60_epoch_lr_15_epoch${i}"
    # AFILE="sup_RN50_from_gauss_4_for_40_epoch_to_gauss_0_for_20_epoch${i}"
    # AFILE="sup_RN50_from_gauss_4_for_40_epoch_to_gauss_0_for_20_epoch${i}"
    # AFILE="shape_bias_gauss_4_for_30_epoch_to_gauss_0_for_30_lr_15_epoch${i}"
    AFILE="gauss_4_sup_RN50_gauss_0_for_60_epoch${i}"
    # AFILE="gauss_6_sup_RN50_gauss_6_for_60_epoch${i}"

    STIM="16_class_IN"

    # RESUME="/data/blurry_vision_sup_RN50/supervised_resnet50_gauss_1_for_60_epoch/outmodel/checkpoint_supervised_resnet50_gauss_1_for_60_epoch_epoch${i}.pth.tar"
    # RESUME="/data/blurry_vision_sup_RN50/supervised_resnet50_from_gauss_4_for_30_epoch_to_gauss_0_for_30_TRAIN2/outmodel/checkpoint_supervised_resnet50_from_gauss_4_for_30_epoch_to_gauss_0_for_30_TRAIN2_epoch${i}.pth.tar"
    # RESUME="/data/blurry_vision_sup_RN50/supervised_resnet50_gauss_4_for_30_epoch_to_gauss_0_for_30_lr_15/outmodel/checkpoint_supervised_resnet50_gauss_4_for_30_epoch_to_gauss_0_for_30_lr_15_epoch${i}.pth.tar"
    # RESUME="/data/blurry_vision_sup_RN50/supervised_resnet50_from_gauss_4_for_40_epoch_to_gauss_0_for_20/outmodel/checkpoint_supervised_resnet50_from_gauss_4_for_40_epoch_to_gauss_0_for_20_epoch${i}.pth.tar"
    # RESUME="/data/blurry_vision_sup_RN50/supervised_resnet50_gauss_2_for_60_epoch/outmodel/checkpoint_supervised_resnet50_gauss_2_for_60_epoch${i}.pth.tar"
    
 
    # RESUME="/data/blurry_vision_sup_RN50/supervised_resnet50_gauss_6_for_60_epoch/outmodel/checkpoint_supervised_resnet50_gauss_6_for_60_epoch_epoch${i}.pth.tar"
    RESUME="/data/blurry_vision_sup_RN50/supervised_resnet50_gauss_0_for_60_epoch/outmodel/checkpoint_supervised_resnet50_gauss_0_for_60_epoch${i}.pth.tar"
    # cd blurry_vision/pytorch_untrained_models/imagenet/

    # REGUALR TEST IMAGES
    # supervised_resnet50_gauss_4_for_60_epoch_lr_15

    # --gauss ${GAUSS} \
    # ${PYTHON} main_val_shape_bias_df_blur.py --a ${MODEL} \
    # ${PYTHON} main_val_shape_bias_df.py --a ${MODEL} \
    # --print-freq ${PRINT} \
    # --batch-size ${BATCH} \
    # --resume ${RESUME} \
    # --save_accuracy_file ${AFILE} \
    # --test_result_stimuli_name ${STIM} \
    # --evaluate \
    # ${DIR}


    # BLURRED TEST IMAGES

    GAUSS=4
    # GAUSS=6
    ${PYTHON} main_val_shape_bias_df_blur.py --a ${MODEL} \
    --print-freq ${PRINT} \
    --batch-size ${BATCH} \
    --resume ${RESUME} \
    --save_accuracy_file ${AFILE} \
    --test_result_stimuli_name ${STIM} \
    --gauss ${GAUSS} \
    --evaluate \
    ${DIR}


    ((i=i+5))

done


