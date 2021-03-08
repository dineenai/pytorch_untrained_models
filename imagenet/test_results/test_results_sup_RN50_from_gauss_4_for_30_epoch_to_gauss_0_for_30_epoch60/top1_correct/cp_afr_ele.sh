while read -r line
do
    echo "$line"
    cp /home/ainedineen/blurry_vision/pytorch_untrained_models/imagenet/val_images_tex/gw_shrk/val/images/$line  /home/ainedineen/blurry_vision/pytorch_untrained_models/imagenet/correctly_identified_val_images_tex_style_transfer/sup_RN50_from_gauss_4_for_30_epoch_to_gauss_0_for_30_epoch60/gw_shrk/$line --recursive
done < /home/ainedineen/blurry_vision/pytorch_untrained_models/imagenet/test_results/test_results_sup_RN50_from_gauss_4_for_30_epoch_to_gauss_0_for_30_epoch60/top1_correct/top1_correct_sup_RN50_from_gauss_4_for_30_epoch_to_gauss_0_for_30_epoch60_gw_shrk.txt  
