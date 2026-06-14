#!/usr/bin/bash

GPU="0"

model_names=("dinov2" "vit" "swinv2" "dinov3_vit")
model_paths=("facebook/dinov2-giant-imagenet1k-1-layer" "google/vit-base-patch16-224" "microsoft/swinv2-base-patch4-window16-256" "facebook/dinov3-vitl16-pretrain-lvd1689m")

drop_nums=("4" "8")
prune_methods=("layer_drop_attn" "layer_drop_mlp" "layer_drop_all" "block_drop")
num_runs=5
batch_sizes="128"

# Place Your Huggingface Token below and uncomment
#export HUGGINGFACE_TOKEN=

for i in "${!model_names[@]}"
do
    model_name="${model_names[$i]}"
    model_path="${model_paths[$i]}"

    if [ ! -d "./${model_name}_model_original" ]; then
        echo "Downloading model..."
        git lfs install
        git clone https://user:$HUGGINGFACE_TOKEN@huggingface.co/${model_path}
        mv $(basename ${model_path}) ./"${model_name}"_model_original
        echo "Model downloaded to ./${model_name}_model_original"
    else
        echo "Model already exists at ./${model_name}_model_original, skipping download"
    fi

    echo "Benchmarking original model: ${model_name}"
    save_file="./results_speed/${model_name}_original_speed.csv"
    baseline_file="./results_speed/${model_name}_original_speed.csv"

    CUDA_VISIBLE_DEVICES=$GPU python src/benchmark_vision_speed.py \
        --model_path ./${model_name}_model_original \
        --batch_sizes "${batch_sizes}" \
        --num_iterations 100 \
        --num_runs ${num_runs} \
        --save_file ${save_file}


    if [ ! -d "./${model_name}_model" ]; then
        echo "Downloading model..."
        git lfs install
        git clone https://user:$HUGGINGFACE_TOKEN@huggingface.co/${model_path}
        mv $(basename ${model_path}) ./"${model_name}"_model
    fi

    for prune_method in "${prune_methods[@]}"
    do
        for drop_num in "${drop_nums[@]}"
        do
            cfg_path=./results_prune/"${model_name}"-"${prune_method}"-discrete-drop"${drop_num}"/checkpoint/config.json
            if [ ! -f "$cfg_path" ]; then
                echo "Config not found: $cfg_path - skipping"
                continue
            fi

            cp -f "$cfg_path" ./"${model_name}"_model/config.json
            cp ./results_prune/"${model_name}"-"${prune_method}"-discrete-drop"${drop_num}"/checkpoint/*.py ./"${model_name}"_model/
            echo "Benchmarking: ${model_name} - ${prune_method} - drop${drop_num}"
            save_file="./results_speed/${model_name}_drop${drop_num}_${prune_method}_speed.csv"

            CUDA_VISIBLE_DEVICES=$GPU python src/benchmark_vision_speed.py \
                --model_path ./${model_name}_model \
                --batch_sizes "${batch_sizes}" \
                --num_iterations 100 \
                --num_runs ${num_runs} \
                --save_file ${save_file} \
                --baseline_file ${baseline_file}
        done
    done
done
