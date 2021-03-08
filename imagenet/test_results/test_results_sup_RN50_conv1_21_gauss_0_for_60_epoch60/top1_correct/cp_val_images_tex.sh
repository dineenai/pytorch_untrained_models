while read -r line
do
    echo "$line"
    cp /home/ainedineen/blurry_vision/pytorch_untrained_models/imagenet/val_images_tex/starfsh/val/images/$line  /home/ainedineen/blurry_vision/pytorch_untrained_models/imagenet/correctly_identified_val_images_tex_style_transfer/sup_RN50_conv1_21_gauss_0_for_60_epoch60/starfsh/$line --recursive
done < /home/ainedineen/blurry_vision/pytorch_untrained_models/imagenet/test_results/test_results_sup_RN50_conv1_21_gauss_0_for_60_epoch60/top1_correct/top1_correct_sup_RN50_conv1_21_gauss_0_for_60_epoch60_starfsh.txt  
