import os
import re
import sys
import torch
import argparse

from pathlib import Path
from accelerate import Accelerator
from trainer import HealingTrainer
from config import HealingConfig
from utils import (load_model_for_healing, get_dropped_layers_info, save_healed_model)

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from llmtuner.compression.prune.utils import is_vision_model
from llmtuner.data.utils import is_vision_dataset
from llmtuner.compression.prune.io import load_cached_similarities


def parse_args():
    parser = argparse.ArgumentParser(description="Heal dropped layers in compressed models")
    # Model paths
    parser.add_argument("--dropped_model_path", type=str, required=True, help="Path to the dropped model directory")
    parser.add_argument("--original_model_path", type=str, required=True, help="Path to the original model directory")
    parser.add_argument("--healed_model_path", type=str, required=True, help="Path where the healed model will be saved")
    
    # Dataset parameters
    parser.add_argument("--dataset", type=str, required=True, help="Dataset name for training")
    parser.add_argument("--dataset_dir", type=str, required=True, help="Directory containing datasets")
    parser.add_argument("--max_length", type=int, required=True, help="Maximum sequence length")
    parser.add_argument("--n_train_samples", type=int, required=True, help="Number of training samples (use -1 for all)")
    parser.add_argument("--val_split_ratio", type=float, required=True, help="Validation split ratio (e.g., 0.2 for 20%)")
    
    # Training hyperparameters
    parser.add_argument("--num_epochs", type=int, required=True, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, required=True, help="Training batch size per GPU")
    parser.add_argument("--learning_rate", type=float, required=True, help="Learning rate for optimizer")
    parser.add_argument("--weight_decay", type=float, required=True, help="Weight decay (L2 regularization)")
    
    # LoRA parameters
    parser.add_argument("--lora_rank", type=int, required=True, help="Rank of LoRA decomposition")
    parser.add_argument("--lora_alpha", type=float, required=True, help="LoRA alpha scaling factor")
    
    # Gradient parameters
    parser.add_argument("--gradient_accumulation_steps", type=int, required=True, help="Number of gradient accumulation steps")
    
    # Evaluation parameters
    parser.add_argument("--max_val_batches", type=int, required=True, help="Maximum validation batches per epoch")
    
    # DataLoader parameters
    parser.add_argument("--num_workers", type=int, required=True, help="Number of data loading workers")
    
    # Model info for output filename
    parser.add_argument("--drop_num", type=str, default=None, help="Number of layers dropped (for output filename)")
    parser.add_argument("--prune_method", type=str, default=None, help="Pruning method used (for output filename)")
    
    return parser.parse_args()


def load_original_similarities(dropped_model_path: str, accelerator: Accelerator) -> dict:
    model_folder = None
    for part in Path(dropped_model_path).parts:
        if "layer_drop" in part or "block_drop" in part:
            model_folder = part
            break

    match = re.match(r'^(.+?)-(layer_drop_(\w+)|block_drop)-', model_folder)
    if not match:
        raise ValueError(f"Cannot parse folder: {model_folder}")

    model_name = match.group(1)
    is_block_drop = "block_drop" in model_folder
    target_layer = match.group(3) if match.group(3) else "block"
    cache_dir = Path("./results_prune/cache")
    if not cache_dir.exists():
        raise FileNotFoundError(f"Cache not found: {cache_dir}")

    def load_sim_file(pattern):
        files = list(cache_dir.glob(pattern))
        if not files:
            raise FileNotFoundError(f"No files matching: {pattern}")
        sim = load_cached_similarities(str(files[0]), 'cpu', accelerator)
        if sim is None:
            raise ValueError(f"Failed to load: {files[0].name}")
        return {i: v.item() for i, v in enumerate(sim) if v != float('-inf')}

    if target_layer == 'all':
        attn = load_sim_file(f"{model_name}-layer_drop_all_attn-*.pt")
        mlp = load_sim_file(f"{model_name}-layer_drop_all_mlp-*.pt")
        similarities = {**attn, **{k + len(attn): v for k, v in mlp.items()}}
    elif is_block_drop:
        similarities = load_sim_file(f"{model_name}-block_drop-*.pt")
    else:
        similarities = load_sim_file(f"{model_name}-layer_drop_{target_layer}-*.pt")
    accelerator.print(f"✅ Loaded {len(similarities)} similarities")

    return similarities


def main():
    args = parse_args()
    accelerator = Accelerator()

    accelerator.print("="*80)
    accelerator.print("GPU CONFIGURATION")
    accelerator.print(f"Accelerator state: {accelerator.state}")
    accelerator.print(f"Number of processes: {accelerator.num_processes}")
    accelerator.print(f"Device: {accelerator.device}")
    accelerator.print(f"CUDA device count: {torch.cuda.device_count()}")
    accelerator.print("="*80)
    
    # Load model
    model, tokenizer_or_processor, dropped_config = load_model_for_healing(args.dropped_model_path, args.original_model_path, accelerator)
    is_vision_model_type = is_vision_model(model)
    is_vision_dataset_type = is_vision_dataset(args.dataset)

    # Validate model and dataset compatibility
    if is_vision_model_type and not is_vision_dataset_type:
        accelerator.print(f"❌ ERROR: Vision model requires vision dataset, got: {args.dataset}")
        exit(1)
    if not is_vision_model_type and is_vision_dataset_type:
        accelerator.print(f"❌ ERROR: Language model cannot use vision dataset, got: {args.dataset}")
        exit(1)

    train_dataset = args.dataset
    val_dataset = args.dataset
    healing_config = HealingConfig(
        dropped_model_path=args.dropped_model_path,
        output_dir=args.healed_model_path,
        num_epochs=args.num_epochs,
        batch_size=args.batch_size,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        dataset_dir=args.dataset_dir,
        n_train_samples=None if args.n_train_samples == -1 else args.n_train_samples,
        val_split_ratio=args.val_split_ratio,
        max_length=args.max_length,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        lora_rank=args.lora_rank,
        lora_alpha=args.lora_alpha,
        lora_dropout=0.0,
        adam_beta1=0.9,
        adam_beta2=0.999,
        adam_epsilon=1e-8,
        scheduler_type="cosine",
        lr_eta_min=0.0,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        max_grad_norm=1.0,
        max_val_batches=args.max_val_batches,
        num_workers=args.num_workers
    )
    
    accelerator.print("="*80)
    accelerator.print("MODEL HEALING PROCESS")
    accelerator.print(f"Model type: {'Vision' if is_vision_model_type else 'Language'}")
    accelerator.print(f"Input model: {healing_config.dropped_model_path}")
    accelerator.print(f"Dataset: {healing_config.train_dataset}")
    accelerator.print(f"Dataset directory: {healing_config.dataset_dir}")
    accelerator.print(f"Training epochs: {healing_config.num_epochs}")
    accelerator.print(f"Batch size: {healing_config.batch_size}")
    accelerator.print("="*80)
    

    drop_attn_indices, drop_mlp_indices = get_dropped_layers_info(dropped_config)
    accelerator.print(f"\nDropped attention layers: {drop_attn_indices}")
    accelerator.print(f"Dropped MLP layers: {drop_mlp_indices}")
    if not drop_attn_indices and not drop_mlp_indices:
        accelerator.print("ERROR: No dropped layers found in model configuration!")
        return
    accelerator.wait_for_everyone()

    accelerator.print("Loading layer similarities...")
    layer_similarities = load_original_similarities(healing_config.dropped_model_path, accelerator)
    for layer_idx in drop_attn_indices + drop_mlp_indices:
        if layer_idx not in layer_similarities:
            accelerator.print(f"ERROR: Missing similarities")
            return


    accelerator.print("Initializing trainer...")
    trainer = HealingTrainer(model, tokenizer_or_processor, dropped_config, healing_config, accelerator, 
                            drop_num=args.drop_num, prune_method=args.prune_method)
    accelerator.print("Starting training...")
    result = trainer.train(drop_attn_indices, drop_mlp_indices, layer_similarities)
    
    # Save healed model
    accelerator.print("Saving healed model...")
    save_healed_model(
        model=trainer.model, tokenizer_or_processor=tokenizer_or_processor,
        output_dir=healing_config.output_dir, accelerator=accelerator)
    
    accelerator.print("="*80)
    accelerator.print("HEALING COMPLETE!")
    if result["final_metrics"]:
        accelerator.print(f"Final validation perplexity: {result['final_metrics'].get('val_perplexity', 'N/A'):.4f}")
    accelerator.print(f"Hyperparameters: {result['hyperparams']}")
    accelerator.print(f"LoRA Config: rank={healing_config.lora_rank}, alpha={healing_config.lora_alpha}, dropout={healing_config.lora_dropout}")
    accelerator.print("="*80)
    accelerator.print("\n\n")


if __name__ == "__main__":
    main()
