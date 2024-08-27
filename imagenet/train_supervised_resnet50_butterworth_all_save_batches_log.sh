#!/bin/bash
#
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=12
#SBATCH -J all_train_log
#SBATCH --output=/data2/blurry_vision_sup_RN50/supervised_resnet50_bp_butter_all_batches_log_for_60_epoch_iter-1/logs/slurm-%j.out
#SBATCH --error=/data2/blurry_vision_sup_RN50/supervised_resnet50_bp_butter_all_batches_log_for_60_epoch_iter-1/logs/slurm-%j.err



ITER=1 
BAND='all'
OUTPTH="/data2/blurry_vision_sup_RN50/supervised_resnet50_bp_butter_${BAND}_batches_log_for_60_epoch_iter-${ITER}"

# mkdir -p ${OUTPTH}/logs
# mkdir ${OUTPTH}/outmodel


DIR="/data/ILSVRC2012"
OUTFOLDER="${OUTPTH}/outmodel"
PYTHON="/opt/anaconda3/envs/blurry_vision/bin/python"
MODEL='resnet50'
CPKT="supervised_resnet50_bp_butter_${BAND}_batches_for_60_epoch"
EPOCHS=3
SAVE=1
ACCPATH="/home/ainedineen/blurry_vision/pytorch_untrained_models/imagenet/bandpass_analysis_butterworth/batched_accuracy"
PRINTFREQ=100
BATCHNOPATH="/home/ainedineen/blurry_vision/pytorch_untrained_models/imagenet/log_batches.csv"
RESUME="${OUTFOLDER}/checkpoint_supervised_resnet50_bp_butter_all_batches_for_60_epoch_epoch0_complete_log_iter-1.pth.tar"


${PYTHON} main_general_copy_mmk_pared_back_save_in_epoch_save_log_batches.py --a ${MODEL} \
--model_path ${OUTFOLDER} \
--epochs ${EPOCHS} \
--cpkt_name ${CPKT} \
--save_freq ${SAVE} \
--print-freq ${PRINTFREQ} \
--path_acc ${ACCPATH} \
--iter ${ITER} \
--batches_to_save ${BATCHNOPATH} \
--resume ${RESUME} \
${DIR}