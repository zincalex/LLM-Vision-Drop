import os
import sys
import json
import torch
import deepspeed
from pathlib import Path
from peft import PeftModel
from accelerate import Accelerator
from typing import List, Tuple
from transformers import AutoConfig, AutoTokenizer, AutoModel, AutoModelForCausalLM, AutoModelForImageClassification, AutoImageProcessor

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from llmtuner.compression.prune.utils import is_vision_model


def load_model_for_healing(dropped_model_path: str, original_model_path: str, accelerator: Accelerator):
    config_json_path = Path(dropped_model_path) / "config.json"
    accelerator.print(f"\nReading drop indices from: {config_json_path}")
    with open(config_json_path, 'r') as f:
        dropped_config_dict = json.load(f)
    
    # Extract drop indices
    drop_attn_list = dropped_config_dict.get('drop_attn_list', [])
    drop_mlp_list = dropped_config_dict.get('drop_mlp_list', [])
    model_type = dropped_config_dict.get('model_type', 'unknown')
    accelerator.print(f"  Model type: {model_type}")
    accelerator.print(f"  Dropped attention layers: {drop_attn_list}")
    accelerator.print(f"  Dropped MLP layers: {drop_mlp_list}")

    class DropConfig:
        def __init__(self, drop_attn_list, drop_mlp_list, model_type):
            self.drop_attn_list = drop_attn_list
            self.drop_mlp_list = drop_mlp_list
            self.model_type = model_type
    
    dropped_config = DropConfig(drop_attn_list, drop_mlp_list, model_type)
    config_kwargs = {"trust_remote_code": True, "local_files_only": True,
        "cache_dir": None, "revision": "main", "token": None, "attn_implementation": "eager"}
    
    accelerator.print(f"\nLoading original model from: {original_model_path}")
    original_config = AutoConfig.from_pretrained(original_model_path, **config_kwargs)
    is_vision = is_vision_model(original_config)

    if is_vision:
        auto_cls = AutoModel if getattr(original_config, "model_type", None) == "dinov3_vit" else AutoModelForImageClassification
        model = auto_cls.from_pretrained(original_model_path, torch_dtype=torch.bfloat16, **config_kwargs)
        processor = AutoImageProcessor.from_pretrained(original_model_path, **config_kwargs)
    else:
        model = AutoModelForCausalLM.from_pretrained(original_model_path, torch_dtype=torch.bfloat16, **config_kwargs)
        try:
            processor = AutoTokenizer.from_pretrained(original_model_path, use_fast=False, **config_kwargs)
        except:
            processor = AutoTokenizer.from_pretrained(original_model_path, use_fast=True, **config_kwargs)

        if processor.pad_token is None: # Set pad token if needed
            processor.pad_token = processor.eos_token

    model.config.use_cache = False

    accelerator.print(f"   Model class: {model.__class__.__name__}")
    accelerator.print(f"   Model module: {model.__class__.__module__}")
    accelerator.print("=" * 80)
    accelerator.print("\n")

    return model, processor, dropped_config


def get_dropped_layers_info(config) -> Tuple[List[int], List[int]]:
    drop_attn_indices = list(config.drop_attn_list) if hasattr(config, 'drop_attn_list') and config.drop_attn_list else []
    drop_mlp_indices = list(config.drop_mlp_list) if hasattr(config, 'drop_mlp_list') and config.drop_mlp_list else []

    return drop_attn_indices, drop_mlp_indices


def save_healed_model(model, tokenizer_or_processor, output_dir: str, accelerator: Accelerator):
    accelerator.wait_for_everyone()

    unwrapped_model = accelerator.unwrap_model(model)
    is_peft = isinstance(unwrapped_model, PeftModel)

    if is_peft:
        lora_params = [p for n, p in unwrapped_model.named_parameters() if 'lora' in n and p.requires_grad]
        using_deepspeed = lora_params and hasattr(lora_params[0], 'ds_id')

        if using_deepspeed:
            with deepspeed.zero.GatheredParameters(lora_params, modifier_rank=0):
                if accelerator.is_main_process:
                    os.makedirs(output_dir, exist_ok=True)
                    unwrapped_model.save_pretrained(output_dir, safe_serialization=True)
        else:
            if accelerator.is_main_process:
                os.makedirs(output_dir, exist_ok=True)
                unwrapped_model.save_pretrained(output_dir, safe_serialization=True)
    else:
        if accelerator.is_main_process:
            os.makedirs(output_dir, exist_ok=True)
            unwrapped_model.save_pretrained(output_dir)

    accelerator.wait_for_everyone()

    if accelerator.is_main_process:
        tokenizer_or_processor.save_pretrained(output_dir)
        accelerator.print(f"✅ Healed model saved to: {output_dir}")
