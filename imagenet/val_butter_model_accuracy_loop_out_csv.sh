#!/bin/bash
#
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=12
#SBATCH -J model_accuracy
#SBATCH --output=/home/ainedineen/blurry_vision/pytorch_untrained_models/imagenet/model_accuracy/logs/slurm-%j.out
#SBATCH --error=/home/ainedineen/blurry_vision/pytorch_untrained_models/imagenet/model_accuracy/logs/slurm-%j.err

# starting epoch
i=1
t=1


until [ $i -gt $t ]
do

    echo i: $i
    
    

    TRAINBAND='low' #Will set path in RESUME

    TESTBAND='all'
    DIR='/data/ILSVRC2012/'
    
    # TESTBAND='low'
    # DIR="/data2/ILSVRC2012/butterworth/cut-0.055-0.15_order-3.0_npad-40/${TESTBAND}" # DIR='/data/ILSVRC2012/'

    echo DIR: $DIR

    PYTHON="/opt/anaconda3/envs/blurry_vision/bin/python"
    PRINT=100
    BATCH=100
    MODEL='resnet50'


    RESUME="/data/blurry_vision_sup_RN50/supervised_resnet50_bp_butter_${TRAINBAND}_for_60_epoch/outmodel/checkpoint_supervised_resnet50_bp_butter_${TRAINBAND}_for_60_epoch_epoch${i}.pth.tar"
    echo RESUME: $RESUME
    AFILE="supervised_resnet50_bp_butter_train-${TRAINBAND}_test-${TESTBAND}" 
    APATH="/home/ainedineen/blurry_vision/pytorch_untrained_models/imagenet/bandpass_analysis_butterworth/model_accuracy_csv"

    ${PYTHON} main_val_model_accuracy_out_csv.py --a ${MODEL} \
    --print-freq ${PRINT} \
    --batch-size ${BATCH} \
    --resume ${RESUME} \
    --save_accuracy_path ${APATH} \
    --save_accuracy_file ${AFILE} \
    --evaluate \
    ${DIR}
  
    # # ((i=i+5))
    ((i=i+1))
done
