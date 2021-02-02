
#!/bin/bash
#
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=12
#SBATCH -J blurry_train_supervised_ResNet50_elephantcat
#SBATCH --output=/home/ainedineen/blurry_vision/pytorch_untrained_models/imagenet/supervised_resnet50/logs/slurm-%j.out
#SBATCH --error=/home/ainedineen/blurry_vision/pytorch_untrained_models/imagenet/supervised_resnet50/logs/slurm-%j.err


DIR='/data/ILSVRC2012/'
PYTHON="/opt/anaconda3/envs/blurry_vision/bin/python"
MODEL='resnet50'


${PYTHON} main.py --a ${MODEL} \
${DIR}
