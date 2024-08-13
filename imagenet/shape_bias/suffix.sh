
# CAUTION: use echo to trial a command and prevent disasters!!

# # remove the end of the file: here the end to be removed was shape_bias_
# for file in *.txtshape_bias_; do
#     mv -- "$file" "${file%%shape_bias_}"
# done

# # Remove the start of a file name: here the start to be removed was 'shape_bias_'
# for file in shape_bias_shape_bias*; do
#     mv -i "$file" "${file#shape_bias_}"
# done

# # Substitute a portion of a filename:here shape_bias_sup_RN50_fgauss --> shape_bias_sup_RN50_gauss
# for file in shape_bias_sup_RN50_fgauss*; do
#     mv $file ${file//shape_bias_sup_RN50_fgauss/shape_bias_sup_RN50_gauss} ;
# done

# # Substitute a portion of a filename:
# for file in shape_bias_shape_bias_gauss_4_for_30_epoch_to_gauss_0_for_30_lr_15*; do
#     echo mv $file ${file//shape_bias_shape_bias_gauss_4_for_30_epoch_to_gauss_0_for_30_lr_15/shape_bias_gauss_4_sup_RN50_gauss_4_for_30_epoch_to_gauss_0_for_30_lr_15} ;
# done

    
# shape_bias_shape_bias_gauss_4_for_30_epoch_to_gauss_0_for_30_lr_15
# shape_bias_gauss_4_sup_RN50_gauss_4_for_30_epoch_to_gauss_0_for_30_lr_15


