import os
import sys
import shutil
import torch
import argparse

from pathlib import Path
from transformers import AutoConfig, AutoTokenizer, AutoModel, AutoModelForCausalLM, AutoModelForImageClassification, AutoImageProcessor
from peft import PeftModel

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from llmtuner.compression.prune.utils import is_vision_model


def merge_lora_adapters(peft_model_path: str, original_model_path: str, output_dir: str):
    print(f"Loading base model from: {original_model_path}")

    config = AutoConfig.from_pretrained(original_model_path, trust_remote_code=True)
    is_vision = is_vision_model(config)

    print(f"Loading base model on cuda:0...")
    if is_vision:
        auto_cls = AutoModel if getattr(config, "model_type", None) == "dinov3_vit" else AutoModelForImageClassification
        base_model = auto_cls.from_pretrained(
            original_model_path, torch_dtype=torch.bfloat16,
            trust_remote_code=True, device_map="cuda:0")
    else:
        base_model = AutoModelForCausalLM.from_pretrained(
            original_model_path, torch_dtype=torch.bfloat16,
            trust_remote_code=True, device_map="cuda:0")

    print(f"Loading PEFT adapters from: {peft_model_path}")
    peft_model = PeftModel.from_pretrained(base_model, peft_model_path)

    print("Merging LoRA adapters into base model...")
    merged_model = peft_model.merge_and_unload()

    os.makedirs(output_dir, exist_ok=True)
    merged_model.save_pretrained(output_dir)

    original_config_path = Path(original_model_path) / "config.json"
    output_config_path = Path(output_dir) / "config.json"
    if original_config_path.exists():
        shutil.copy2(original_config_path, output_config_path)
        print("Replaced config.json with original")

    if is_vision:
        processor = AutoImageProcessor.from_pretrained(original_model_path, trust_remote_code=True)
        processor.save_pretrained(output_dir)
    else:
        tokenizer = AutoTokenizer.from_pretrained(original_model_path, use_fast=False, trust_remote_code=True)
        tokenizer.save_pretrained(output_dir)

    for file_pattern in ["configuration_*.py", "modeling_*.py"]:
        for file_path in Path(original_model_path).glob(file_pattern):
            shutil.copy2(file_path, Path(output_dir) / file_path.name)

    print(f"Merge complete! Saved to: {output_dir}")


def parse_args():
    parser = argparse.ArgumentParser(description="Merge LoRA adapters into base model")
    parser.add_argument("--peft_model_path", type=str, required=True)
    parser.add_argument("--original_model_path", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    return parser.parse_args()


def main():
    args = parse_args()

    print("=" * 80)
    print("MERGING LORA ADAPTERS INTO BASE MODEL")
    print("=" * 80)

    merge_lora_adapters(peft_model_path=args.peft_model_path, original_model_path=args.original_model_path,
                        output_dir=args.output_dir)
    print(f"Merged model saved to: {args.output_dir}")
    print("=" * 80)


if __name__ == "__main__":
    main()
