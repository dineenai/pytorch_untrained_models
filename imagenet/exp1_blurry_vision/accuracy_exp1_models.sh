#!/bin/bash
#
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=12
#SBATCH -J shape_bias_exp1_models
#SBATCH --output=/home/ainedineen/blurry_vision/pytorch_untrained_models/imagenet/exp1_blurry_vision/slurm_logs/slurm-%j.out
#SBATCH --error=/home/ainedineen/blurry_vision/pytorch_untrained_models/imagenet/exp1_blurry_vision/slurm_logs/slurm-%j.err


source /foundcog/pyenv3.8/bin/activate


# DIR='/home/ainedineen/blurry_vision/texture-vs-shape/stimuli/' # Stimuli directory
DIR='/data/ILSVRC2012/'
PYTHON="/opt/anaconda3/envs/blurry_vision/bin/python"
# PRINT=1
# BATCH=1
PRINT=100
BATCH=100
MODEL='resnet50'
# STIM="16_class_IN"
STIM="imagenet_val"
# GAUSS=6

# Define list of Gaussian levels
# gauss_list=(0 0.5 1 1.5 2 3 4 6)
gauss_list=(6)

# # ALL MODELS
# # Read models into an array
# model_paths=()
# while IFS= read -r line || [[ -n "$line" ]]; do
#     model_paths+=("$line")
#     done < /home/ainedineen/blurry_vision/pytorch_untrained_models/imagenet/exp1_blurry_vision/exp1_models_remaining.txt
# # done < /home/ainedineen/blurry_vision/pytorch_untrained_models/imagenet/exp1_blurry_vision/exp1_model_list.txt

# SINGLE MODEL
model_paths=(/data/blurry_vision_sup_RN50/supervised_resnet50_gauss_1pt5_for_60_epoch/outmodel/checkpoint_supervised_resnet50_gauss_1pt5_for_60_epoch_epoch60.pth.tar)


# Total number of models
num_models=${#model_paths[@]}


# Iterate over the models
for (( i=0; i<num_models; i++ )); do
    # modelpth=${model_paths[i]}
    # echo "Processing model: $modelpth"
    RESUME=${model_paths[i]}

    # Extract train gauss from the model path
    if [[ "$RESUME" =~ gauss_([0-9]+)(pt([0-9]+))? ]]; then
        train_gauss="${BASH_REMATCH[1]}"
        if [[ -n "${BASH_REMATCH[3]}" ]]; then
            train_gauss="${train_gauss}.${BASH_REMATCH[3]}"
        fi
    else
        echo "Warning: Could not extract train gauss from path: $RESUME"
        train_gauss="unknown"
    fi

    
    # iterate over each Gaussian level
    for GAUSS in "${gauss_list[@]}"; do
        NAME="train-${train_gauss}_test-${GAUSS}"

        echo "Running model ${RESUME} with Gaussian level ${GAUSS}"
        echo "Train Gaussian: ${train_gauss}, Test Gaussian: ${GAUSS} Name: ${NAME}"

        # srun
        ${PYTHON} main_val_shape_bias_df_blur_mmk.py --a ${MODEL} \
        --print-freq ${PRINT} \
        --batch-size ${BATCH} \
        --resume ${RESUME} \
        --test_result_stimuli_name ${STIM} \
        --gauss ${GAUSS} \
        --save_accuracy_file ${NAME} \
        --save_accuracy_path "/home/ainedineen/blurry_vision/pytorch_untrained_models/imagenet/exp1_blurry_vision/exp1_accuracy" \
        --evaluate \
        ${DIR}

    done
done

    # srun --exclusive --ntasks=1 --nodes=1 --job-name="$modelpth" python /home/ainedineen/bootstrapping_blurry/dnn_comparisons_blurry_Mar25.py \
    #     --calc True \

# /data/blurry_vision_sup_RN50/supervised_resnet50_gauss_6_for_60_epoch_mmk/outmodel/checkpoint_supervised_resnet50_gauss_6_for_60_epoch_mmk_epoch60.pth.tar
# --save_accuracy_file ${AFILE} \