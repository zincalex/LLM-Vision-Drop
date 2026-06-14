import os
import sys
import json
import torch
import random
import torch.nn as nn

from PIL import Image
from tqdm import tqdm
from pathlib import Path
from torch.optim import AdamW
from accelerate import Accelerator
from typing import Dict, List, Any, Tuple
from peft import LoraConfig, get_peft_model
from transformers import default_data_collator
from torch.utils.data import DataLoader, Dataset
from torch.optim.lr_scheduler import CosineAnnealingLR

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))


class HealingTrainerVM:
    def __init__(self, model, processor, config, healing_config, accelerator: Accelerator, drop_num=None, prune_method=None):
        self.model = model
        self.processor = processor
        self.config = config
        self.healing_config = healing_config
        self.accelerator = accelerator
        self.drop_num = drop_num
        self.prune_method = prune_method

        # Enable gradient checkpointing, skip for SwinV2
        model_type = getattr(self.model.config, "model_type", "unknown")
        if hasattr(self.model, 'gradient_checkpointing_enable') and model_type != "swinv2":
            try:
                self.model.gradient_checkpointing_enable()
            except:
                pass

        self.setup_datasets()
        accelerator.print(f"✅ Trainer initialized")


    def replace_classification_head(self):
        dataset_path = os.path.join(self.healing_config.dataset_dir, f"{self.healing_config.train_dataset}.json")
        with open(dataset_path, 'r') as f:
            data = json.load(f)
        dataset_classes = len(set(item["label"] for item in data))

        model_classes = getattr(self.model.config, 'num_labels', None)
        if model_classes != dataset_classes:
            self.accelerator.print(f"Replacing classification head: {model_classes} → {dataset_classes}")
            self.model.config.num_labels = dataset_classes
            self.model.num_labels = dataset_classes

            config = self.model.config
            if hasattr(config, 'hidden_size'):
                hidden_size = config.hidden_size
            elif hasattr(config, 'embed_dim'):
                hidden_size = config.embed_dim
            elif hasattr(config, 'hidden_sizes'):
                hidden_size = config.hidden_sizes[-1]
            else:
                raise ValueError("Could not determine hidden size")

            if hasattr(self.model, 'dinov2') or 'dinov2' in self.model.__class__.__name__.lower():
                classifier_input_size = hidden_size * 2
            else:
                classifier_input_size = hidden_size

            model_dtype = next(self.model.parameters()).dtype
            self.model.classifier = nn.Linear(classifier_input_size, dataset_classes, dtype=model_dtype)
            self.accelerator.print(f"  ✅ Head replaced: {classifier_input_size} → {dataset_classes} (FROZEN during healing)")


    def apply_lora(self, drop_attn_indices: List[int], drop_mlp_indices: List[int]) -> int:
        self.accelerator.print("\n" + "=" * 80)
        self.accelerator.print("APPLYING LoRA ADAPTERS")

        model_type = getattr(self.model.config, "model_type", "unknown")
        target_modules = []
        layers_to_transform = []

        if model_type == "swinv2":
            target_modules = self.build_swinv2_target_modules(drop_attn_indices, drop_mlp_indices)
            layers_to_transform = None
        else:
            if drop_attn_indices:
                target_modules.extend(["query", "key", "value"])
                layers_to_transform.extend(drop_attn_indices)
            if drop_mlp_indices:
                if model_type == "dinov2":
                    target_modules.extend(["weights_in", "weights_out"])
                elif model_type == "dinov3_vit":
                    target_modules.extend(["up_proj", "down_proj"])
                elif model_type == "vit":
                    target_modules.extend(["intermediate.dense", "output.dense"])
                layers_to_transform.extend(drop_mlp_indices)

        if not target_modules:
            self.accelerator.print("❌ No target modules")
            return 0

        if layers_to_transform:
            layers_to_transform = sorted(list(set(layers_to_transform)))

        lora_config = LoraConfig(
            r=self.healing_config.lora_rank, lora_alpha=self.healing_config.lora_alpha,
            target_modules=target_modules,
            layers_to_transform=layers_to_transform if layers_to_transform else None,
            lora_dropout=self.healing_config.lora_dropout, bias="none", task_type=None)

        self.model = get_peft_model(self.model, lora_config)
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        self.accelerator.print(f"LoRA applied ✅ Trainable parameters: {trainable_params:,}")
        self.accelerator.print("=" * 80 + "\n")

        return trainable_params


    def build_swinv2_target_modules(self, drop_attn_indices: List[int], drop_mlp_indices: List[int]) -> List[str]:
        depths = self.model.config.depths
        stage_starts = [0]
        for depth in depths:
            stage_starts.append(stage_starts[-1] + depth)

        def global_to_stage_block(global_idx):
            for stage_idx in range(len(depths)):
                if stage_starts[stage_idx] <= global_idx < stage_starts[stage_idx + 1]:
                    return stage_idx, global_idx - stage_starts[stage_idx]
            raise ValueError(f"Global block index {global_idx} out of range")

        target_modules = []
        if drop_attn_indices:
            for global_idx in drop_attn_indices:
                stage_idx, local_block = global_to_stage_block(global_idx)
                for sub in ["query", "key", "value"]:
                    target_modules.append(
                        f"{self.model.base_model_prefix}.encoder.layers.{stage_idx}.blocks.{local_block}.attention.self.{sub}")

        if drop_mlp_indices:
            for global_idx in drop_mlp_indices:
                stage_idx, local_block = global_to_stage_block(global_idx)
                for sub in ["intermediate.dense", "output.dense"]:
                    target_modules.append(
                        f"{self.model.base_model_prefix}.encoder.layers.{stage_idx}.blocks.{local_block}.{sub}")

        self.accelerator.print(f"Built {len(target_modules)} explicit module paths for SwinV2")

        return target_modules


    def setup_datasets(self):
        dataset_path = os.path.join(self.healing_config.dataset_dir, f"{self.healing_config.train_dataset}.json")
        self.train_dataset = self.create_dataset(dataset_path, "train")
        self.val_dataset = self.create_dataset(dataset_path, "val")
        self.accelerator.print(f"Training samples: {len(self.train_dataset)}, Validation samples: {len(self.val_dataset)}")


    def create_dataset(self, data_path: str, split: str):
        class VisionDataset(Dataset):
            def __init__(self, data_path, processor, split, total_samples=None, val_split_ratio=0.2):
                with open(data_path, 'r') as f:
                    all_data = json.load(f)
                train_data = [item for item in all_data if "/test/" not in item["image"]]
                if total_samples and total_samples > 0:
                    train_data = train_data[:total_samples]
                random.seed(2120824)
                shuffled = train_data.copy()
                random.shuffle(shuffled)
                split_idx = int(len(shuffled) * (1.0 - val_split_ratio))
                self.data = shuffled[:split_idx] if split == "train" else shuffled[split_idx:]
                self.processor = processor
                self.base_dir = Path(data_path).parent

            def __len__(self):
                return len(self.data)

            def __getitem__(self, idx):
                item = self.data[idx]
                image = Image.open(self.base_dir / item["image"]).convert("RGB")
                inputs = self.processor(images=image, return_tensors="pt")
                pixel_values = inputs["pixel_values"].squeeze(0).to(torch.bfloat16)
                return {"pixel_values": pixel_values, "labels": torch.tensor(item["label"], dtype=torch.long)}

        return VisionDataset(data_path, self.processor, split,
                             self.healing_config.n_train_samples, self.healing_config.val_split_ratio)


    def create_dataloaders(self) -> Tuple[DataLoader, DataLoader]:
        collator = default_data_collator
        train_dl = DataLoader(self.train_dataset, batch_size=self.healing_config.batch_size,
                              shuffle=True, collate_fn=collator, num_workers=self.healing_config.num_workers, pin_memory=True)
        val_dl = DataLoader(self.val_dataset, batch_size=self.healing_config.batch_size,
                            shuffle=False, collate_fn=collator, num_workers=self.healing_config.num_workers, pin_memory=True)
        return train_dl, val_dl


    def extract_layer_index(self, param_name: str) -> int:
        parts = param_name.split('.')

        # SwinV2: layers.X.blocks.Y -> global index
        if 'layers' in parts and 'blocks' in parts:
            try:
                layers_idx = parts.index('layers')
                blocks_idx = parts.index('blocks')
                if layers_idx + 1 < len(parts) and blocks_idx + 1 < len(parts):
                    stage_idx = int(parts[layers_idx + 1])
                    local_block = int(parts[blocks_idx + 1])
                    if hasattr(self.model.config, 'depths'):
                        depths = self.model.config.depths
                        stage_starts = [0]
                        for d in depths:
                            stage_starts.append(stage_starts[-1] + d)
                        return stage_starts[stage_idx] + local_block
            except (ValueError, IndexError):
                pass

        # Standard: layers.X or blocks.X or encoder.layer.X
        for i, part in enumerate(parts):
            if part in ['layers', 'blocks', 'layer'] and i + 1 < len(parts):
                try:
                    return int(parts[i + 1])
                except ValueError:
                    continue
        return None


    def build_lora_param_map(self, drop_attn_indices: List[int], drop_mlp_indices: List[int]) -> Dict[int, List[torch.nn.Parameter]]:
        layer_params = {}
        for name, param in self.model.named_parameters():
            if not param.requires_grad or 'lora' not in name:
                continue
            layer_idx = self.extract_layer_index(name)
            if layer_idx is None:
                continue
            if layer_idx not in drop_attn_indices and layer_idx not in drop_mlp_indices:
                continue
            if layer_idx not in layer_params:
                layer_params[layer_idx] = []
            layer_params[layer_idx].append(param)
        return layer_params


    def compute_similarity_reg_loss(self, layer_params: Dict[int, List[torch.nn.Parameter]], layer_similarities: Dict[int, float],
                                    reg_weight: float) -> torch.Tensor:
        reg_loss = torch.tensor(0.0, device=self.accelerator.device)
        for layer_idx, params in layer_params.items():
            sim = layer_similarities.get(layer_idx, 0.5)
            weight = 1.0 - sim
            for p in params:
                reg_loss = reg_loss + weight * torch.sum(p ** 2)
        return reg_weight * reg_loss


    def evaluate(self, dataloader: DataLoader) -> Dict[str, float]:
        self.model.eval()
        total_loss = 0.0
        num_batches = 0
        with torch.no_grad():
            for batch_idx, batch in enumerate(dataloader):
                if batch_idx >= self.healing_config.max_val_batches:
                    break
                try:
                    outputs = self.model(pixel_values=batch["pixel_values"], labels=batch["labels"])
                    total_loss += outputs.loss.item()
                    num_batches += 1
                except Exception as e:
                    self.accelerator.print(f"Evaluate error on batch {batch_idx}: {e}")
                    continue

        if num_batches == 0:
            return {"loss": float('inf'), "perplexity": float('inf')}
        avg_loss = total_loss / num_batches
        return {"loss": avg_loss, "perplexity": torch.exp(torch.tensor(avg_loss)).item()}


    def train(self, drop_attn_indices: List[int], drop_mlp_indices: List[int],
              layer_similarities: Dict[int, float]) -> Dict[str, Any]:
        lr = self.healing_config.learning_rate
        reg_weight = self.healing_config.reg_weight
        epochs = self.healing_config.num_epochs
        model_type = getattr(self.model.config, "model_type", "model")

        self.accelerator.print(f"Training: lr={lr}, reg_weight={reg_weight}, epochs={epochs}")
        self.accelerator.print(f"Model type: {model_type}")

        try:
            self.replace_classification_head()
            self.apply_lora(drop_attn_indices, drop_mlp_indices)

            # Build LoRA param map for regularization
            lora_param_map = self.build_lora_param_map(drop_attn_indices, drop_mlp_indices)

            # Print similarity weighting info
            self.accelerator.print("\nSimilarity-weighted regularization (in loss):")
            for layer_idx in sorted(lora_param_map.keys()):
                sim = layer_similarities.get(layer_idx, 0.5)
                n_params = sum(p.numel() for p in lora_param_map[layer_idx])
                self.accelerator.print(
                    f"  Layer {layer_idx}: sim={sim:.4f} → reg_weight_eff={reg_weight * (1.0 - sim):.6f} ({n_params:,} elements)")

            total_elements = sum(p.numel() for params in lora_param_map.values() for p in params)
            self.accelerator.print(f"Total: {len(lora_param_map)} layers, {total_elements:,} trainable elements\n")

            trainable_params = [p for p in self.model.parameters() if p.requires_grad]
            optimizer = AdamW(trainable_params, lr=lr, weight_decay=0.0,
                              betas=(self.healing_config.adam_beta1, self.healing_config.adam_beta2),
                              eps=self.healing_config.adam_epsilon)

            train_dataloader, val_dataloader = self.create_dataloaders()
            total_steps = len(train_dataloader) * epochs

            if self.healing_config.scheduler_type == "cosine":
                scheduler = CosineAnnealingLR(optimizer, T_max=total_steps, eta_min=self.healing_config.lr_eta_min)
            else:
                raise ValueError(f"Unsupported scheduler: {self.healing_config.scheduler_type}")

            # Prepare with accelerator (DDP, no DeepSpeed)
            self.model, optimizer, train_dataloader, val_dataloader, scheduler = self.accelerator.prepare(
                self.model, optimizer, train_dataloader, val_dataloader, scheduler)

        except Exception as e:
            self.accelerator.print(f"❌ Setup failed: {e}")
            return {"hyperparams": {}, "final_metrics": {}, "training_history": []}

        training_history = []
        log_file = None
        if self.accelerator.is_main_process:
            dataset_name = self.healing_config.train_dataset
            log_file = f"output_healing_{model_type}_drop{self.drop_num}_{self.prune_method}_{dataset_name}.out"
            with open(log_file, 'w') as f:
                f.write("=" * 80 + "\n")
                f.write("TRAINING METRICS LOG (Vision-only, DDP, Custom Loss Regularization)\n")
                f.write("=" * 80 + "\n")
                f.write(f"Model: {self.healing_config.dropped_model_path}\n")
                f.write(f"Output: {self.healing_config.output_dir}\n")
                f.write(f"Dataset: {self.healing_config.train_dataset}\n")
                f.write(f"Training samples: {len(self.train_dataset)}\n")
                f.write(f"Validation samples: {len(self.val_dataset)}\n")
                f.write(f"Learning Rate: {lr}\n")
                f.write(f"Reg Weight (lambda_base): {reg_weight}\n")
                f.write(f"Optimizer Weight Decay: 0.0\n")
                f.write(f"Epochs: {epochs}\n")
                f.write(f"Batch Size: {self.healing_config.batch_size}\n")
                f.write(f"LoRA Rank: {self.healing_config.lora_rank}\n")
                f.write(f"LoRA Alpha: {self.healing_config.lora_alpha}\n")
                f.write(f"Scheduler: {self.healing_config.scheduler_type}\n")
                f.write("=" * 80 + "\n\n")

        best_val_loss = float('inf')
        best_epoch = 0
        best_model_state = None

        for epoch in range(epochs):
            self.model.train()
            epoch_task_loss = 0.0
            epoch_reg_loss = 0.0
            num_batches = 0

            pbar = tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{epochs}") if self.accelerator.is_main_process else train_dataloader

            for batch in pbar:
                try:
                    outputs = self.model(pixel_values=batch["pixel_values"], labels=batch["labels"])
                    task_loss = outputs.loss

                    # Custom similarity-weighted L2 regularization
                    reg_loss = self.compute_similarity_reg_loss(lora_param_map, layer_similarities, reg_weight)
                    loss = task_loss + reg_loss

                    self.accelerator.backward(loss)
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad()

                    epoch_task_loss += task_loss.item()
                    epoch_reg_loss += reg_loss.item()
                    num_batches += 1

                    if self.accelerator.is_main_process and isinstance(pbar, tqdm):
                        current_lr = scheduler.get_last_lr()[0]
                        pbar.set_postfix({
                            'task': f'{task_loss.item():.4f}',
                            'reg': f'{reg_loss.item():.4f}',
                            'total': f'{loss.item():.4f}',
                            'lr': f'{current_lr:.2e}'
                        })
                except Exception as e:
                    self.accelerator.print(f"Warning: Step failed: {e}")
                    continue

            if self.accelerator.is_main_process and isinstance(pbar, tqdm):
                pbar.close()

            # Validation
            try:
                val_metrics = self.evaluate(val_dataloader)
            except Exception:
                val_metrics = {"loss": float('inf'), "perplexity": float('inf')}

            epoch_metrics = {
                "epoch": epoch + 1,
                "train_task_loss": epoch_task_loss / max(num_batches, 1),
                "train_reg_loss": epoch_reg_loss / max(num_batches, 1),
                "train_total_loss": (epoch_task_loss + epoch_reg_loss) / max(num_batches, 1),
                "val_loss": val_metrics["loss"],
                "val_perplexity": val_metrics["perplexity"],
            }
            training_history.append(epoch_metrics)

            current_val_loss = val_metrics["loss"]
            is_best = current_val_loss < best_val_loss

            if is_best:
                best_val_loss = current_val_loss
                best_epoch = epoch + 1
                lora_state_dict = {name: param.detach().clone()
                                   for name, param in self.model.named_parameters()
                                   if 'lora' in name and param.requires_grad}
                best_model_state = {
                    'epoch': best_epoch,
                    'val_loss': best_val_loss,
                    'val_perplexity': val_metrics["perplexity"],
                    'lora_state_dict': lora_state_dict
                }
                if self.accelerator.is_main_process:
                    self.accelerator.print(f"  🌟 New best model! (val_loss: {best_val_loss:.4f})")

            if self.accelerator.is_main_process:
                message = (f"Epoch {epoch+1}/{epochs}: "
                          f"task_loss={epoch_metrics['train_task_loss']:.4f}, "
                          f"reg_loss={epoch_metrics['train_reg_loss']:.6f}, "
                          f"val_loss={val_metrics['loss']:.4f}, "
                          f"val_ppl={val_metrics['perplexity']:.4f}")
                if current_val_loss == best_val_loss:
                    message += " ⭐ BEST"
                self.accelerator.print(message)
                if log_file:
                    with open(log_file, 'a') as f:
                        f.write(message + "\n")

        # Summary
        if self.accelerator.is_main_process:
            self.accelerator.print("\n" + "=" * 80)
            self.accelerator.print("TRAINING COMPLETE")
            self.accelerator.print(f"Best model from epoch {best_epoch} with val_loss: {best_val_loss:.4f}")
            self.accelerator.print(f"Final model from epoch {epochs} with val_loss: {training_history[-1]['val_loss']:.4f}")
            if best_epoch != epochs:
                improvement = training_history[-1]['val_loss'] - best_val_loss
                self.accelerator.print(f"⚠️  Final model is {improvement:.4f} worse. Restoring best from epoch {best_epoch}...")
            else:
                self.accelerator.print(f"✅ Final model is the best model!")
            self.accelerator.print("=" * 80 + "\n")
            if log_file:
                with open(log_file, 'a') as f:
                    f.write(f"\nBest model: Epoch {best_epoch}, Val Loss: {best_val_loss:.4f}\n")

        # Restore best model if needed
        if best_epoch != epochs and best_model_state is not None:
            if 'lora_state_dict' in best_model_state:
                self.model.load_state_dict(best_model_state['lora_state_dict'], strict=False)
                if self.accelerator.is_main_process:
                    self.accelerator.print("✅ Best model restored from memory")

        if self.accelerator.is_main_process and log_file:
            self.accelerator.print(f"✅ Training log saved to: {log_file}")

        return {
            "hyperparams": {
                "lr": lr,
                "reg_weight": reg_weight,
                "optimizer_weight_decay": 0.0,
                "lora_rank": self.healing_config.lora_rank,
                "lora_alpha": self.healing_config.lora_alpha,
                "lora_dropout": self.healing_config.lora_dropout,
                "scheduler_type": self.healing_config.scheduler_type
            },
            "final_metrics": training_history[-1] if training_history else {},
            "best_metrics": best_model_state if best_model_state else {},
            "training_history": training_history
        }
