#supervised_resnet50_from_gauss_4_for_30_epoch_to_gauss_0_for_30
#!/bin/bash
#
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=12
#SBATCH -J blurry_val_supervised_ResNet50_elephantcat
#SBATCH --output=/home/ainedineen/blurry_vision/pytorch_untrained_models/imagenet/supervised_resnet50/evaluate/logs/slurm-%j.out
#SBATCH --error=/home/ainedineen/blurry_vision/pytorch_untrained_models/imagenet/supervised_resnet50/evaluate/logs/slurm-%j.err


DIR='/home/ainedineen/blurry_vision/pytorch_untrained_models/imagenet/correctly_identified_val_images_tex_style_transfer/sup_RN50_from_gauss_4_for_30_epoch_to_gauss_0_for_30_epoch60/ST_ly_bfly_tex_collie/'
PYTHON="/opt/anaconda3/envs/blurry_vision/bin/python"
PRINT=1
BATCH=1
MODEL='resnet50'
AFILE="sup_RN50_from_gauss_4_for_30_epoch_to_gauss_0_for_30_epoch60"
STIM="correctly_identified_val_images_tex_style_transfer_ST_ly_bfly_tex_collie"
RESUME="/data/blurry_vision_sup_RN50/supervised_resnet50_from_gauss_4_for_30_epoch_to_gauss_0_for_30/outmodel/checkpoint_supervised_resnet50_from_gauss_4_for_30_epoch_to_gauss_0_for_30_epoch60.pth.tar"



${PYTHON} main_val.py --a ${MODEL} \
--print-freq ${PRINT} \
--batch-size ${BATCH} \
--resume ${RESUME} \
--save_accuracy_file ${AFILE} \
--test_result_stimuli_name ${STIM} \
--evaluate \
${DIR}
