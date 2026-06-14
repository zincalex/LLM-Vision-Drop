import os
import sys
import json
import torch
import random
import torch.nn as nn

from tqdm import tqdm
from PIL import Image
from pathlib import Path
from torch.optim import AdamW
from accelerate import Accelerator
from typing import Dict, List, Any, Tuple
from peft import LoraConfig, get_peft_model
from transformers import default_data_collator
from torch.utils.data import DataLoader, Dataset
from torch.optim.lr_scheduler import CosineAnnealingLR

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from llmtuner.compression.prune.utils import is_vision_model


class HealingTrainer:
    def __init__(self, model, processor, config, healing_config, accelerator: Accelerator, drop_num=None, prune_method=None):
        self.model = model
        self.processor = processor
        self.config = config
        self.healing_config = healing_config
        self.accelerator = accelerator
        self.drop_num = drop_num
        self.prune_method = prune_method
        
        # Enable gradient checkpointing, but skip for SwinV2
        # SwinV2's patch_embeddings returns tuple which breaks PEFT's make_inputs_require_grads hook
        model_type = getattr(self.model.config, "model_type", "unknown")
        if hasattr(self.model, 'gradient_checkpointing_enable') and model_type != "swinv2":
            try:
                self.model.gradient_checkpointing_enable()
            except:
                pass
        
        self.setup_datasets()
        accelerator.print(f"✅ Trainer initialized")

    
    def replace_classification_head(self):
        is_vision = is_vision_model(self.model)
        if not is_vision:
            return  # Only for vision models
        
        # Get dataset number of classes
        dataset_path = os.path.join(self.healing_config.dataset_dir, f"{self.healing_config.train_dataset}.json")
        with open(dataset_path, 'r') as f:
            data = json.load(f)
        dataset_classes = len(set(item["label"] for item in data))
        
        # Get model number of classes
        model_classes = getattr(self.model.config, 'num_labels', None)
        if model_classes != dataset_classes:
            self.accelerator.print(f"Replacing classification head...")
            self.accelerator.print(f"  Current model classes: {model_classes}")
            self.accelerator.print(f"  Dataset classes: {dataset_classes}")

            self.model.config.num_labels = dataset_classes
            self.model.num_labels = dataset_classes
            
            # Get hidden size
            config = self.model.config
            if hasattr(config, 'hidden_size'):
                hidden_size = config.hidden_size
            elif hasattr(config, 'embed_dim'):
                hidden_size = config.embed_dim
            elif hasattr(config, 'hidden_sizes'):
                hidden_size = config.hidden_sizes[-1]
            else:
                raise ValueError(f"Could not determine hidden size for model")
            
            # Check if DINOv2 (uses hidden_size * 2)
            if hasattr(self.model, 'dinov2') or 'dinov2' in self.model.__class__.__name__.lower():
                classifier_input_size = hidden_size * 2
                self.accelerator.print(f"  DINOv2 detected: classifier input = {classifier_input_size}")
            else:
                classifier_input_size = hidden_size
            
            # Replace classifier
            model_dtype = next(self.model.parameters()).dtype
            self.model.classifier = nn.Linear(classifier_input_size, dataset_classes, dtype=model_dtype)
            
            self.accelerator.print(f"  ✅ Classification head replaced: {classifier_input_size} → {dataset_classes}")
            self.accelerator.print(f"  ⚠️  Note: Classifier will remain FROZEN during healing")
            self.accelerator.print(f"           (It will be trained from scratch during benchmarking)")


    def apply_lora(self, drop_attn_indices: List[int], drop_mlp_indices: List[int]) -> int:
        self.accelerator.print("\n" + "=" * 80)
        self.accelerator.print("APPLYING LoRA ADAPTERS")
        
        is_vision = is_vision_model(self.model)
        model_type = getattr(self.model.config, "model_type", "llama")
        
        target_modules = []
        layers_to_transform = []
        
        if is_vision:
            # For SwinV2, we need to build explicit module paths due to hierarchical structure
            if model_type == "swinv2":
                target_modules = self.build_swinv2_target_modules(drop_attn_indices, drop_mlp_indices)
                layers_to_transform = None  # Not used when target_modules are explicit paths
            else:
                if drop_attn_indices:
                    target_modules.extend(["query", "key", "value"])
                    layers_to_transform.extend(drop_attn_indices)
                if drop_mlp_indices:
                    if model_type == "dinov2":
                        target_modules.extend(["weights_in", "weights_out"])
                        layers_to_transform.extend(drop_mlp_indices)
                    elif model_type == "dinov3_vit":
                        target_modules.extend(["up_proj", "down_proj"])
                        layers_to_transform.extend(drop_mlp_indices)
                    elif model_type == "vit":
                        target_modules.extend(["intermediate.dense", "output.dense"])
                        layers_to_transform.extend(drop_mlp_indices)
        else: # Language models
            if drop_attn_indices:
                target_modules.extend(["q_proj", "k_proj", "v_proj", "o_proj"])
                layers_to_transform.extend(drop_attn_indices)
            if drop_mlp_indices:
                target_modules.extend(["gate_proj", "up_proj", "down_proj"])
                layers_to_transform.extend(drop_mlp_indices)
        
        if not target_modules:
            self.accelerator.print("❌ No target modules")
            return 0

        if layers_to_transform:
            layers_to_transform = sorted(list(set(layers_to_transform)))
        
        if model_type == "swinv2":
            self.accelerator.print(f"Target modules: {len(target_modules)} explicit paths")
            self.accelerator.print(f"  Sample: {target_modules[:3]}")
        else:
            self.accelerator.print(f"Target modules: {target_modules}")
            if layers_to_transform:
                self.accelerator.print(f"Layers to transform: {layers_to_transform}")

        lora_config = LoraConfig(
            r=self.healing_config.lora_rank, lora_alpha=self.healing_config.lora_alpha,
            target_modules=target_modules, 
            layers_to_transform=layers_to_transform if layers_to_transform else None,
            lora_dropout=self.healing_config.lora_dropout, bias="none",
            task_type="CAUSAL_LM" if not is_vision else None)
        
        # Count total modules for verification
        if model_type != "swinv2":
            all_modules = list(self.model.named_modules())
            matching_modules = [name for name, _ in all_modules if any(target in name for target in target_modules)]
            self.accelerator.print(f"Total modules in model: {len(all_modules)}")
            self.accelerator.print(f"Modules matching target: {len(matching_modules)}")
        
        self.model = get_peft_model(self.model, lora_config)
        
        # Count trainable parameters (LoRA adapters only, classifier is frozen)
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        self.accelerator.print(f"LoRA applied ✅")
        self.accelerator.print(f"Trainable parameters: {trainable_params:,} (LoRA adapters only)")
        self.accelerator.print("=" * 80 + "\n")
        return trainable_params
    
    
    def build_swinv2_target_modules(self, drop_attn_indices: List[int], drop_mlp_indices: List[int]) -> List[str]:
        depths = self.model.config.depths  # Number of blocks per stage
        
        # Build stage info: cumulative block counts
        stage_starts = [0]
        for depth in depths:
            stage_starts.append(stage_starts[-1] + depth)
        
        self.accelerator.print(f"SwinV2 stage structure: {depths} blocks per stage")
        self.accelerator.print(f"Stage boundaries: {stage_starts}")
        
        # Convert global indices to (stage, local_block) pairs
        def global_to_stage_block(global_idx):
            for stage_idx in range(len(depths)):
                if stage_starts[stage_idx] <= global_idx < stage_starts[stage_idx + 1]:
                    local_block = global_idx - stage_starts[stage_idx]
                    return stage_idx, local_block
            raise ValueError(f"Global block index {global_idx} out of range")
        
        target_modules = []
        if drop_attn_indices:
            attn_submodules = ["query", "key", "value"]
            for global_idx in drop_attn_indices:
                stage_idx, local_block = global_to_stage_block(global_idx)
                for submodule in attn_submodules:
                    path = f"{self.model.base_model_prefix}.encoder.layers.{stage_idx}.blocks.{local_block}.attention.self.{submodule}"
                    target_modules.append(path)
                self.accelerator.print(f"  Global block {global_idx} → stage {stage_idx}, local block {local_block}")

        if drop_mlp_indices:
            mlp_submodules = ["intermediate.dense", "output.dense"]
            for global_idx in drop_mlp_indices:
                stage_idx, local_block = global_to_stage_block(global_idx)
                for submodule in mlp_submodules:
                    path = f"{self.model.base_model_prefix}.encoder.layers.{stage_idx}.blocks.{local_block}.{submodule}"
                    target_modules.append(path)
        
        self.accelerator.print(f"Built {len(target_modules)} explicit module paths for SwinV2")
        return target_modules


    def setup_datasets(self):
        is_vision = is_vision_model(self.model)
        
        if is_vision:
            dataset_path = os.path.join(self.healing_config.dataset_dir, f"{self.healing_config.train_dataset}.json")
            self.train_dataset = self.create_vision_dataset(dataset_path, "train")
            self.val_dataset = self.create_vision_dataset(dataset_path, "val")
        else:
            train_path = os.path.join(self.healing_config.dataset_dir, f"{self.healing_config.train_dataset}.json")
            val_path = os.path.join(self.healing_config.dataset_dir, f"{self.healing_config.val_dataset}.json")
            self.train_dataset = self.create_text_dataset(train_path)
            self.val_dataset = self.create_text_dataset(val_path)
        
        self.accelerator.print(f"Training samples: {len(self.train_dataset)}, Validation samples: {len(self.val_dataset)}")


    def create_vision_dataset(self, data_path: str, split: str):
        class VisionDataset(Dataset):
            def __init__(self, data_path, processor, split, total_samples=None, val_split_ratio=0.2):
                with open(data_path, 'r') as f:
                    all_data = json.load(f)
                
                # Only use training data
                train_data = [item for item in all_data if "/test/" not in item["image"]]
                if total_samples and total_samples > 0:
                    train_data = train_data[:total_samples]

                random.seed(2120824)
                shuffled_train = train_data.copy()
                random.shuffle(shuffled_train)
                
                split_idx = int(len(shuffled_train) * (1.0 - val_split_ratio))
                if split == "train":
                    data = shuffled_train[:split_idx]
                else:  # val
                    data = shuffled_train[split_idx:]
                
                self.data = data
                self.processor = processor
                self.base_dir = Path(data_path).parent
            
            def __len__(self):
                return len(self.data)
            
            def __getitem__(self, idx):
                item = self.data[idx]
                image_path = self.base_dir / item["image"]
                label = item["label"]
                
                image = Image.open(image_path).convert("RGB")
                inputs = self.processor(images=image, return_tensors="pt")
                
                # Convert pixel values to bfloat16 to match model dtype
                pixel_values = inputs["pixel_values"].squeeze(0).to(torch.bfloat16)
                
                return {"pixel_values": pixel_values, "labels": torch.tensor(label, dtype=torch.long)}
        
        max_samples = self.healing_config.n_train_samples
        val_split_ratio = self.healing_config.val_split_ratio
        return VisionDataset(data_path, self.processor, split, max_samples, val_split_ratio)


    def create_text_dataset(self, data_path: str):
        class TextDataset(Dataset):
            def __init__(self, data_path, tokenizer, max_length, max_samples=None):
                with open(data_path, 'r') as f:
                    data = json.load(f)
                
                if max_samples:
                    data = data[:max_samples]
                
                self.data = data
                self.tokenizer = tokenizer
                self.max_length = max_length
            
            def __len__(self):
                return len(self.data)
            
            def __getitem__(self, idx):
                item = self.data[idx]
                text = item.get("text", item.get("prompt", "")) + self.tokenizer.eos_token
                
                tokenized = self.tokenizer(
                    text, max_length=self.max_length, truncation=True,
                    padding="max_length", return_tensors="pt")
                
                # Create labels with -100 for padding tokens (ignored in loss)
                labels = tokenized["input_ids"].squeeze(0).clone()
                labels[labels == self.tokenizer.pad_token_id] = -100
                
                return {
                    "input_ids": tokenized["input_ids"].squeeze(0),
                    "attention_mask": tokenized["attention_mask"].squeeze(0),
                    "labels": labels
                }
        
        max_samples = self.healing_config.n_train_samples if "train" in data_path else None
        return TextDataset(data_path, self.processor, self.healing_config.max_length, max_samples)


    def create_dataloaders(self) -> Tuple[DataLoader, DataLoader]:
        data_collator = default_data_collator

        train_dataloader = DataLoader(
            self.train_dataset, batch_size=self.healing_config.batch_size,
            shuffle=True, collate_fn=data_collator, num_workers=self.healing_config.num_workers,
            pin_memory=True)
        
        val_dataloader = DataLoader(
            self.val_dataset, batch_size=self.healing_config.batch_size,
            shuffle=False, collate_fn=data_collator, num_workers=self.healing_config.num_workers,
            pin_memory=True)
        
        return train_dataloader, val_dataloader


    def evaluate(self, dataloader: DataLoader, is_vision: bool) -> Dict[str, float]:
        self.model.eval()
        total_loss = 0.0
        num_batches = 0
        max_val_batches = self.healing_config.max_val_batches
        
        with torch.no_grad():
            for batch_idx, batch in enumerate(dataloader):
                if batch_idx >= max_val_batches:
                    break
                    
                try:
                    if is_vision:
                        outputs = self.model(pixel_values=batch["pixel_values"], labels=batch["labels"])
                    else:
                        outputs = self.model(
                            input_ids=batch["input_ids"], attention_mask=batch["attention_mask"],
                            labels=batch["labels"])
                    
                    total_loss += outputs.loss.item()
                    num_batches += 1
                except Exception as e:
                    self.accelerator.print(f"Evaluate error on batch {batch_idx}: {e}")
                    continue
        
        if num_batches == 0:
            return {"loss": float('inf'), "perplexity": float('inf')}
        
        avg_loss = total_loss / num_batches
        perplexity = torch.exp(torch.tensor(avg_loss)).item()
        
        return {"loss": avg_loss, "perplexity": perplexity}


    def extract_layer_index(self, param_name: str) -> int:
        parts = param_name.split('.')
        
        # SwinV2 case: layers.X.blocks.Y -> convert to global index
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
                        for depth in depths:
                            stage_starts.append(stage_starts[-1] + depth)
                        
                        global_idx = stage_starts[stage_idx] + local_block
                        return global_idx
            except (ValueError, IndexError):
                pass

        if 'layers.' in param_name or 'blocks.' in param_name or 'encoder.layer.' in param_name:
            for i, part in enumerate(parts):
                if part in ['layers', 'blocks', 'layer'] and i + 1 < len(parts):
                    try:
                        return int(parts[i + 1])
                    except ValueError:
                        continue
        
        return None


    def create_similarity_weighted_param_groups(self, drop_attn_indices: List[int], drop_mlp_indices: List[int], layer_similarities: Dict[int, float],
                                                base_weight_decay: float) -> List[Dict]:
        self.layer_weight_info = {}
        
        # Organize LoRA parameters by layer
        layer_params = {}  # {layer_idx: [param1, param2, ...]}
        
        for name, param in self.model.named_parameters():
            if not param.requires_grad:
                continue
            
            # Only LoRA parameters (classifier is frozen)
            if 'lora' not in name:
                continue
            
            layer_idx = self.extract_layer_index(name)
            if layer_idx is None:
                continue
            
            # Only include dropped layers
            if layer_idx not in drop_attn_indices and layer_idx not in drop_mlp_indices:
                continue
            
            if layer_idx not in layer_params:
                layer_params[layer_idx] = []
            layer_params[layer_idx].append(param)
        
        # Create parameter groups with weighted decay
        param_groups = []
        
        # Add LoRA parameter groups (one per layer)
        for layer_idx in sorted(layer_params.keys()):
            params = layer_params[layer_idx]
            sim = layer_similarities.get(layer_idx, 0.5)
            
            # Inverse weighting: high similarity → low weight decay
            weight = 1.0 - sim
            layer_wd = base_weight_decay * weight
            
            param_groups.append({
                'params': params,
                'weight_decay': layer_wd
            })
            
            # Store for logging
            self.layer_weight_info[layer_idx] = {'similarity': sim, 'weight_decay': layer_wd,
                                                 'num_params': len(params), 'num_elements': sum(p.numel() for p in params)}
        return param_groups


    def train(self, drop_attn_indices: List[int], drop_mlp_indices: List[int], layer_similarities: Dict[int, float]) -> Dict[str, Any]:
        lr = self.healing_config.learning_rate
        base_weight_decay = self.healing_config.weight_decay
        epochs = self.healing_config.num_epochs
        is_vision = is_vision_model(self.model)

        model_type = getattr(self.model.config, "model_type", "model")
        
        self.accelerator.print(f"Training: lr={lr}, base_weight_decay={base_weight_decay}, epochs={epochs}")
        self.accelerator.print(f"Model type: {model_type}")
        
        try:
            self.replace_classification_head()

            # Apply LoRA
            self.apply_lora(drop_attn_indices, drop_mlp_indices)
            
            # Create parameter groups with similarity-weighted weight decay
            param_groups = self.create_similarity_weighted_param_groups(
                drop_attn_indices, drop_mlp_indices, layer_similarities, base_weight_decay)
            
            # Print similarity weighting info - show ALL layers
            self.accelerator.print("\nSimilarity-weighted parameter groups:")
            all_dropped = sorted(set(drop_attn_indices + drop_mlp_indices))
            for layer_idx in all_dropped:  # Show all layers, not just first 5
                info = self.layer_weight_info.get(layer_idx, {})
                self.accelerator.print(
                    f"  Layer {layer_idx}: sim={info.get('similarity', 0):.4f} → "
                    f"weight_decay={info.get('weight_decay', 0):.6f} "
                    f"({info.get('num_params', 0)} params, {info.get('num_elements', 0):,} elements)"
                )

            total_elements = sum(info['num_elements'] for info in self.layer_weight_info.values())
            self.accelerator.print(f"Total: {len(param_groups)} parameter groups, {total_elements:,} trainable parameters\n")
            
            # Create optimizer with parameter groups
            optimizer = AdamW(param_groups, lr=lr,
                            betas=(self.healing_config.adam_beta1, self.healing_config.adam_beta2),
                            eps=self.healing_config.adam_epsilon)
            train_dataloader, val_dataloader = self.create_dataloaders()
            total_steps = len(train_dataloader) * epochs
            
            if self.healing_config.scheduler_type == "cosine":
                scheduler = CosineAnnealingLR(optimizer, T_max=total_steps, eta_min=self.healing_config.lr_eta_min)
            else:
                raise ValueError(f"Unsupported scheduler type: {self.healing_config.scheduler_type}")
            
            # Prepare with accelerator - this wraps the PEFT model
            self.model, optimizer, train_dataloader, val_dataloader, scheduler = self.accelerator.prepare(
                self.model, optimizer, train_dataloader, val_dataloader, scheduler)
            
        except Exception as e:
            self.accelerator.print(f"❌ Setup failed: {e}")
            return {"hyperparams": {}, "final_metrics": {}, "training_history": []}
        
        training_history = []
        log_file = None
        
        # Create log file for training metrics
        if self.accelerator.is_main_process:
            dataset_name = self.healing_config.train_dataset
            log_file = f"output_healing_{model_type}_drop{self.drop_num}_{self.prune_method}_{dataset_name}.out"
            with open(log_file, 'w') as f:
                f.write("=" * 80 + "\n")
                f.write("TRAINING METRICS LOG\n")
                f.write("=" * 80 + "\n")
                f.write(f"Model: {self.healing_config.dropped_model_path}\n")
                f.write(f"Output: {self.healing_config.output_dir}\n")
                f.write(f"Dataset: {self.healing_config.train_dataset}\n")
                f.write(f"Training samples: {len(self.train_dataset)}\n")
                f.write(f"Validation samples: {len(self.val_dataset)}\n")
                f.write(f"Learning Rate: {lr}\n")
                f.write(f"Base Weight Decay: {base_weight_decay}\n")
                f.write(f"Epochs: {epochs}\n")
                f.write(f"Batch Size: {self.healing_config.batch_size}\n")
                f.write(f"Gradient Accumulation Steps: {self.healing_config.gradient_accumulation_steps}\n")
                f.write(f"LoRA Rank: {self.healing_config.lora_rank}\n")
                f.write(f"LoRA Alpha: {self.healing_config.lora_alpha}\n")
                f.write(f"LoRA Dropout: {self.healing_config.lora_dropout}\n")
                f.write(f"Scheduler: {self.healing_config.scheduler_type}\n")
                f.write("=" * 80 + "\n\n")
        
        # Track best model based on validation loss
        best_val_loss = float('inf')
        best_epoch = 0
        best_model_state = None
        for epoch in range(epochs):
            self.model.train()
            epoch_loss = 0.0
            num_batches = 0
            
            if self.accelerator.is_main_process:
                pbar = tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{epochs}")
            else:
                pbar = train_dataloader
            
            for batch in pbar:
                try:   # Forward pass
                    if is_vision:
                        outputs = self.model(pixel_values=batch["pixel_values"], labels=batch["labels"])
                    else:
                        outputs = self.model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"],
                            labels=batch["labels"])
                    
                    loss = outputs.loss
                    self.accelerator.backward(loss)
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad()
                    
                    epoch_loss += loss.item()
                    num_batches += 1
                    
                    if self.accelerator.is_main_process and isinstance(pbar, tqdm):
                        current_lr = scheduler.get_last_lr()[0]
                        avg_loss = epoch_loss / num_batches
                        pbar.set_postfix({
                            'loss': f'{loss.item():.4f}',
                            'avg_loss': f'{avg_loss:.4f}',
                            'lr': f'{current_lr:.2e}'
                        })
                        
                except Exception as e:
                    self.accelerator.print(f"Warning: Step failed: {e}")
                    continue
            
            if self.accelerator.is_main_process and isinstance(pbar, tqdm):
                pbar.close()
            
            # Validation
            try:
                val_metrics = self.evaluate(val_dataloader, is_vision)
            except Exception:
                val_metrics = {"loss": float('inf'), "perplexity": float('inf')}
            
            epoch_metrics = {"epoch": epoch + 1, "train_loss": epoch_loss / max(num_batches, 1),
                             "val_loss": val_metrics["loss"], "val_perplexity": val_metrics["perplexity"]}
            
            training_history.append(epoch_metrics)
            
            # Check if this is the best model so far
            current_val_loss = val_metrics["loss"]
            is_best = current_val_loss < best_val_loss
            if is_best:
                best_val_loss = current_val_loss
                best_epoch = epoch + 1
                
                # Store best model state in memory (LoRA adapters only - lightweight)
                # Keep on GPU
                lora_state_dict = {name: param.detach().clone() 
                                   for name, param in self.model.named_parameters() 
                                   if 'lora' in name and param.requires_grad}
                
                best_model_state = {'epoch': best_epoch, 'val_loss': best_val_loss,
                                    'val_perplexity': val_metrics["perplexity"],'lora_state_dict': lora_state_dict}
                
                if self.accelerator.is_main_process:
                    self.accelerator.print(f"  🌟 New best model! (val_loss: {best_val_loss:.4f})")
            
            if self.accelerator.is_main_process:
                message = (f"Epoch {epoch+1}/{epochs}: "
                          f"train_loss={epoch_metrics['train_loss']:.4f}, "
                          f"val_loss={val_metrics['loss']:.4f}, "
                          f"val_ppl={val_metrics['perplexity']:.4f}")
                if current_val_loss == best_val_loss:
                    message += " ⭐ BEST"
                self.accelerator.print(message)
                
                # Log to file
                if log_file:
                    with open(log_file, 'a') as f:
                        f.write(message + "\n")
        
        # Print best model summary
        if self.accelerator.is_main_process:
            self.accelerator.print("\n" + "=" * 80)
            self.accelerator.print("TRAINING COMPLETE")
            self.accelerator.print("=" * 80)
            self.accelerator.print(f"Best model from epoch {best_epoch} with validation loss: {best_val_loss:.4f}")
            self.accelerator.print(f"Final model from epoch {epochs} with validation loss: {training_history[-1]['val_loss']:.4f}")
            
            if best_epoch != epochs:
                improvement = training_history[-1]['val_loss'] - best_val_loss
                self.accelerator.print(f"⚠️  Note: Final model is {improvement:.4f} worse than best model")
                self.accelerator.print(f"         Restoring best model from epoch {best_epoch}...")
            else:
                self.accelerator.print(f"✅ Final model is the best model!")
            self.accelerator.print("=" * 80 + "\n")
            
            if log_file:
                with open(log_file, 'a') as f:
                    f.write("\n" + "=" * 80 + "\n")
                    f.write(f"Best model: Epoch {best_epoch}, Val Loss: {best_val_loss:.4f}\n")
                    f.write("=" * 80 + "\n")
        
        # Restore best model weights if it's not the final epoch
        if best_epoch != epochs and best_model_state is not None:
            if 'lora_state_dict' in best_model_state:
                # Restore LoRA weights from memory (fast - already on GPU)
                self.model.load_state_dict(best_model_state['lora_state_dict'], strict=False)
                if self.accelerator.is_main_process:
                    self.accelerator.print("✅ Best model restored from memory")
        
        if self.accelerator.is_main_process and log_file:
            self.accelerator.print(f"✅ Training log saved to: {log_file}")
        
        return {
            "hyperparams": {
                "lr": lr,
                "base_weight_decay": base_weight_decay,
                "lora_rank": self.healing_config.lora_rank,
                "lora_alpha": self.healing_config.lora_alpha,
                "lora_dropout": self.healing_config.lora_dropout,
                "adam_beta1": self.healing_config.adam_beta1,
                "adam_beta2": self.healing_config.adam_beta2,
                "scheduler_type": self.healing_config.scheduler_type
            },
            "final_metrics": training_history[-1] if training_history else {},
            "best_metrics": best_model_state if best_model_state else {},
            "training_history": training_history
        }
