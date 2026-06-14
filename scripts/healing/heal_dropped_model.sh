#!/usr/bin/bash

port="21307"
GPUs="0,1,2,3"

model_names=("mistral" "llama2-13b" "dinov2" "vit" "swinv2")
model_paths=("mistralai/Mistral-7B-v0.1" "meta-llama/Llama-2-13b-hf" "facebook/dinov2-giant-imagenet1k-1-layer" "google/vit-base-patch16-224" "microsoft/swinv2-base-patch4-window16-256")

prune_methods=("layer_drop_attn" "layer_drop_mlp" "layer_drop_all" "block_drop")
drop_nums=("10")
drop_method="discrete"

# Dataset parameters
dataset="imagenet-1k_demo"
dataset_dir="./data/imagenet-1k"
max_length=2048
n_train_samples=-1
val_split_ratio=0.2

# Training hyperparameters
num_epochs=20
batch_size=8
learning_rate=2e-3
weight_decay=0.03

# LoRA parameters
lora_rank=8
lora_alpha=16.0

# Gradient parameters
gradient_accumulation_steps=4

# Evaluation parameters
max_val_batches=20

# DataLoader parameters
num_workers=4

# Place Your Huggingface Token below and uncomment
#export HUGGINGFACE_TOKEN=

for i in "${!model_names[@]}"
do
    model_name="${model_names[$i]}"
    model_path="${model_paths[$i]}"
    original_model_dir="./${model_name}_model_original"
    if [ ! -d "$original_model_dir" ]; then
        echo "Downloading original model for healing..."
        git lfs install
        git clone https://user:$HUGGINGFACE_TOKEN@huggingface.co/${model_path}
        mv $(basename ${model_path}) "$original_model_dir"
        echo "Clean model downloaded to $original_model_dir"
    else
        echo "Clean model already exists at $original_model_dir"
    fi

    for prune_method in "${prune_methods[@]}"
    do
        for drop_n in "${drop_nums[@]}"
        do
            folder_name="${model_name}-${prune_method}-${drop_method}-drop${drop_n}"
            dropped_model_path="./results_prune/${folder_name}/checkpoint"
            healed_model_path="./results_prune/${folder_name}/checkpoint_healed_${dataset}"

            echo "Starting model healing for: ${folder_name} with dataset: ${dataset}"
            CUDA_VISIBLE_DEVICES=$GPUs accelerate launch \
              --config_file src/model-healing/deepspeed_config.yaml \
              --main_process_port $port \
              src/model-healing/heal_model.py \
              --dropped_model_path "$dropped_model_path" \
              --original_model_path "$original_model_dir" \
              --healed_model_path "$healed_model_path" \
              --dataset "$dataset" \
              --dataset_dir "$dataset_dir" \
              --max_length $max_length \
              --n_train_samples $n_train_samples \
              --val_split_ratio $val_split_ratio \
              --num_epochs $num_epochs \
              --batch_size $batch_size \
              --learning_rate $learning_rate \
              --weight_decay $weight_decay \
              --lora_rank $lora_rank \
              --lora_alpha $lora_alpha \
              --gradient_accumulation_steps $gradient_accumulation_steps \
              --max_val_batches $max_val_batches \
              --num_workers $num_workers \
              --drop_num "$drop_n" \
              --prune_method "$prune_method"

            
            # Merge LoRA adapters into base model
            echo "Merging LoRA adapters into base model..."
            python src/model-healing/merge_adapters.py \
              --peft_model_path "$healed_model_path" \
              --original_model_path "$original_model_dir" \
              --output_dir "${healed_model_path}_merged"
        done
    done
done
