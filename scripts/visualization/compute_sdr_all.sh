#!/usr/bin/bash

datasets=("imagenet-1k" "zoolake" "cifar10" "lar" "CrossD" "LCZ42")
results_dir="src/visualization/benchmark_vm_results"
speed_dir="src/visualization/speed_vm_results"
output_dir="src/visualization/sdr_results"

speed_files=$(find ${speed_dir} -name "*_drop*_speed.csv" -type f)

for speed_file in $speed_files
do
    filename=$(basename "$speed_file" .csv)
    model=$(echo "$filename" | sed 's/_drop.*//')
    drop_num=$(echo "$filename" | grep -oP 'drop\K[0-9]+')
    prune_method=$(echo "$filename" | sed "s/${model}_drop${drop_num}_//" | sed 's/_speed$//')

    found_results=false
    for dataset in "${datasets[@]}"
    do
        result_file="${results_dir}/${dataset}/output_vision_${model}_drop${drop_num}_${prune_method}_${dataset}.out"
        if [ -f "$result_file" ]; then
            found_results=true
            break
        fi
    done

    python3 src/visualization/compute_sdr.py \
        --model ${model} \
        --prune_method ${prune_method} \
        --drop_num ${drop_num} \
        --datasets ${datasets[@]} \
        --results_dir ${results_dir} \
        --speed_dir ${speed_dir} \
        --output_dir ${output_dir}
done

echo "=========================================="
echo "All SDR computations completed!"
echo "Results saved in ${output_dir}/"
echo "=========================================="
