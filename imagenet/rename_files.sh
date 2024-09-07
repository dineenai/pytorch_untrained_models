PTH="/data2/blurry_vision_sup_RN50/supervised_resnet50_untrained_iterations/outmodel"
echo ${PTH}
for f in ${PTH}/checkpoint_supervised_resnet50*;

do
    # echo ${f}
    # RENAME net-resnet50_BP2-all_batche to net-resnet50_BP2-low_batche
    mv -i -- "$f" "${f//untrained_log/untrained}";
done


# SUCCESS