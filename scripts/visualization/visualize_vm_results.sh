#!/usr/bin/bash

model_names=("dinov2" "vit" "swinv2" "dinov3_vit")
prune_methods=("layer_drop_attn" "layer_drop_mlp" "layer_drop_all" "block_drop")
datasets=("imagenet-1k" "zoolake" "cifar10" "lar" "LCZ42" "CrossD" "Pest" "InfLarynge" "Bark")

results_dir="src/visualization/benchmark_vm_results"
output_dir="src/visualization/results"

for dataset in "${datasets[@]}"
do
    for prune_method in "${prune_methods[@]}"
    do
        echo "Processing: ${dataset} - ${prune_method}"
        drop_nums_found=()
        for model in "${model_names[@]}"
        do
            pattern="${results_dir}/${dataset}/output_vision_${model}_drop*_${prune_method}_${dataset}.out"
            for file in $pattern
            do
                if [ -f "$file" ]; then
                    drop_num=$(echo "$file" | grep -oP 'drop\K[0-9]+')
                    if [[ ! " ${drop_nums_found[@]} " =~ " ${drop_num} " ]]; then
                        drop_nums_found+=("$drop_num")
                    fi
                fi
            done
        done
        
        # Sort drop_nums numerically
        IFS=$'\n' drop_nums_sorted=($(sort -n <<<"${drop_nums_found[*]}"))
        unset IFS
        
        if [ ${#drop_nums_sorted[@]} -eq 0 ]; then
            echo "  ⚠️  No result files found for ${dataset} - ${prune_method}"
            echo ""
            continue
        fi
        
        echo "  Found drop numbers: ${drop_nums_sorted[@]}"
        
        python3 src/visualization/visualize_benchmark_results.py \
            --dataset ${dataset} \
            --prune_method ${prune_method} \
            --model_names ${model_names[@]} \
            --drop_nums ${drop_nums_sorted[@]} \
            --results_dir ${results_dir} \
            --output_dir ${output_dir}
        
        echo ""
    done
done
