from typing import Optional
from dataclasses import dataclass


@dataclass
class HealingConfig:
    # Model paths
    dropped_model_path: str
    output_dir: str
    
    # Training hyperparameters
    num_epochs: int
    batch_size: int
    learning_rate: float
    weight_decay: float
    
    # LoRA configuration
    lora_rank: int
    lora_alpha: float
    
    # Gradient parameters
    gradient_accumulation_steps: int
    
    # Dataset configuration
    train_dataset: str
    val_dataset: str
    dataset_dir: str
    max_length: int
    n_train_samples: Optional[int]
    val_split_ratio: float
    
    # Evaluation parameters
    max_val_batches: int
    
    # DataLoader parameters
    num_workers: int
    
    # Fields with defaults
    lora_dropout: float = 0.0
    adam_beta1: float = 0.9
    adam_beta2: float = 0.999
    adam_epsilon: float = 1e-8
    scheduler_type: str = "cosine"
    lr_eta_min: float = 0.0
    max_grad_norm: float = 1.0
