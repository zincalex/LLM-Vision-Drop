<h1 align="center">Uncovering the Redundancy in Transformers via a Unified Study of Layer Dropping</h1>

<p align="center">
  <a href="https://openreview.net/forum?id=1I7PCbOPfe"><img src="https://img.shields.io/badge/Paper-OpenReview-8A2BE2" alt="OpenReview"></a>
  <a href="https://huggingface.co/collections/LLM-Drop/llm-drop-66dde616140f04eb18424a0a"><img src="https://img.shields.io/badge/Models-Hugging%20Face-F9D371" alt="Hugging Face"></a>
  <img src="https://img.shields.io/badge/TMLR-2026-0B7285" alt="TMLR 2026">
  <img src="https://img.shields.io/badge/Python-3.10+-green" alt="Python 3.10+">
</p>

<p align="center">
  <strong>Alessandro Viespoli</strong>
</p>

<p align="center">
  <em>Based on the original work by <a href="https://shwai-he.github.io/">Shwai He*</a>, <a href="https://s1ghhh.github.io/">Guoheng Sun*</a>, <a href="https://shenzheyu.github.io/">Zheyu Shen</a>, <a href="https://www.ang-li.com/">Ang Li</a> — University of Maryland, College Park</em>
</p>

<p align="center">
  <a href="https://case-lab-umd.github.io/LLM-Drop/">🌐 Project Page</a> •
  <a href="#-news">📰 News</a> •
  <a href="#-installation">⚙️ Installation</a> •
  <a href="#-repository-layout">📦 Layout</a> •
  <a href="#-prepare-models">🧰 Models</a> •
  <a href="#-run-dropping">🚀 Dropping</a> •
  <a href="#-benchmark">📊 Benchmark</a> •
  <a href="#-automated-model-selection">🤖 Model Selection</a> •
  <a href="#-citation">📄 Citation</a>
</p>

This repository extends the official implementation of [**Uncovering the Redundancy in Transformers via a Unified Study of Layer Dropping**](https://openreview.net/forum?id=1I7PCbOPfe) (**TMLR 2026**) with full support for **vision transformers**, multi-dataset benchmarking, and an automated model selection pipeline.

## 📖 Introduction

This project studies architectural redundancy in Transformer models — both LLMs and vision transformers — and provides practical pipelines for:

- **Block Drop** — remove full Transformer blocks (attention + MLP together)
- **Layer Drop** — drop attention or MLP sublayers independently
- **Joint Layer Drop** — drop across both sublayer types simultaneously
- **Vision Transformer Support** — DINOv2, DINOv3 ViT, SwinV2, ViT
- **Automated Model Selection** — grid search over architectures, methods, and drop counts
- **Benchmarking** — task accuracy, inference speed, FLOPs, and the SDR efficiency metric

The dropping pipeline is built on [LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory).

![Layer-Drop.svg](Layer_Drop.svg)

## 📰 News

- **Feb 2026:** Paper published in **Transactions on Machine Learning Research (TMLR)**.
- **May 2025:** 🏆 Awarded the Qualcomm Innovation Fellowship (QIF) North America for *"Less Attention, Much Faster: Toward a Future of Efficiency-Optimized Transformer Architectures."*
- **Nov 2024:** Added support for more model families (Gemma2, Baichuan, DeepSeek, Yi, Solar).
- **Sep 2024:** Released dropped-model checkpoints on [Hugging Face](https://huggingface.co/collections/LLM-Drop/llm-drop-66dde616140f04eb18424a0a).
- **Jun 2024:** Released arXiv preprint and code.

## ⚙️ Installation

```bash
conda create -n llm-drop python=3.10 -y
conda activate llm-drop

git clone https://github.com/CASE-Lab-UMD/LLM-Drop.git
cd LLM-Drop

# Core dropping pipeline
pip install -e .
```

<details>
<summary><strong>Optional: Quantization dependencies (AWQ / GPTQ)</strong></summary>

```bash
cd src/llmtuner/compression/quantization/AutoAWQ
pip install -e .

cd AutoAWQ_kernels
pip install -e .

cd ../../AutoGPTQ
pip install -vvv --no-build-isolation -e .

cd ../../../../../..
```

</details>

## 📦 Repository Layout

```
src/
├── compress.py                   # Entry point for dropping/compression
├── benchmark_speed.py            # LLM inference speed benchmark
├── benchmark_vision_speed.py     # Vision model speed + FLOPs benchmark
├── llmtuner/
│   └── compression/prune/        # Core dropping algorithms (block, layer, joint)
│       └── models/               # Custom dropped-model classes per architecture
├── vm-eval/
│   └── benchmark_vision.py       # Vision evaluation harness (finetune head + test)
├── model-selection/
│   └── pipeline.py               # Automated model selection pipeline
├── model-healing/
│   └── heal_model_vm.py          # LoRA-based recovery after layer dropping
└── visualization/
    ├── compute_sdr.py            # SDR metric computation
    └── visualize_benchmark_results.py

scripts/
├── dropping/                     # Shell scripts for block/layer drop (LLM + vision)
├── benchmark/                    # Evaluation and speed benchmark wrappers
├── model-selection/              # Model selection runner
├── healing/                      # Model healing runner
└── visualization/                # SDR and result plotting
```

## 🧰 Prepare Models

Models are downloaded automatically from Hugging Face on first run via `from_pretrained`. For gated models (e.g. Llama-2), authenticate first:

```bash
huggingface-cli login
```

### Supported architectures

| Domain | Architecture | HuggingFace ID |
|---|---|---|
| Vision | DINOv2 | `facebook/dinov2-giant-imagenet1k-1-layer` |
| Vision | DINOv3 ViT | `facebook/dinov3-vitl16-pretrain-lvd1689m` |
| Vision | SwinV2 | `microsoft/swinv2-base-patch4-window16-256` |
| Vision | ViT | `google/vit-base-patch16-224` |
| LLM | Mistral-7B | `mistralai/Mistral-7B-v0.1` |
| LLM | Llama-2-13B | `meta-llama/Llama-2-13b-hf` |

### Dropped model config

After running the dropping pipeline, `drop_attn_list` and `drop_mlp_list` are written into the model's `config.json`. Example configurations:

```json
// Drop attention layers only
{ "drop_attn_list": [25, 26, 24, 22], "drop_mlp_list": [] }

// Drop MLP layers only
{ "drop_attn_list": [], "drop_mlp_list": [26, 27, 25, 24] }

// Drop full blocks
{ "drop_attn_list": [26, 25, 24, 27], "drop_mlp_list": [26, 25, 24, 27] }
```

Custom model classes are stored under `src/llmtuner/compression/prune/models/` and referenced via `auto_map` in the config.

## 🚀 Run Dropping

```bash
# Vision models
bash scripts/dropping/vision_block_drop.sh
bash scripts/dropping/vision_layer_drop.sh
bash scripts/dropping/vision_layer_drop_joint.sh

# LLMs
bash scripts/dropping/block_drop.sh
bash scripts/dropping/layer_drop.sh
bash scripts/dropping/layer_drop_joint.sh
```

Each script runs in two phases:
1. **Similarity estimation** — computes cosine similarity between layer inputs and outputs on a calibration set, identifies which layers to drop, and saves the config.
2. **Post-dropping** — applies the dropped config to the model checkpoint.

Similarity results are cached as `.pt` files under `results_prune/cache/` so re-running with the same settings skips recomputation.

## 📊 Benchmark

### 🖼️ Vision model evaluation

Evaluates a dropped vision model on a dataset: optionally fine-tunes the classification head, runs inference on the test split, and saves logits/predictions to HDF5. Also prints per-layer execution verification (confirming which attention/MLP sublayers were actually skipped).

```bash
bash scripts/benchmark/benchmark_vm_eval.sh
```

Key arguments (edit the script or call directly):

```bash
CUDA_VISIBLE_DEVICES=0,1 accelerate launch src/vm-eval/benchmark_vision.py \
  --model_name_or_path ./dinov2_model \
  --dataset LCZ42 \
  --dataset_base_dir data \
  --prune_method layer_drop_attn \
  --drop_num 4 \
  --finetune_head \
  --epochs 20 \
  --lr 0.001 \
  --weight_decay 0.03 \
  --batch_size 32 \
  --batch_size_eval 10 \
  --output_file results/dinov2_lcz42_drop4.out
```

### ⚡ Inference speed + FLOPs

Measures throughput (images/s), latency, memory, and FLOPs for vision models. FLOPs computation respects the dropped layers in the config.

```bash
bash scripts/benchmark/benchmark_vm_speed.sh
```

### 📈 SDR metric

The **Speedup Degradation Ratio** (γ = ΔAccuracy / ΔSpeedup) measures accuracy cost per unit of throughput gain. Lower γ = more efficient compression.

```bash
bash scripts/visualization/compute_sdr_all.sh
```

Results are written to `src/visualization/sdr_results/`.

### 🧪 LLM task performance (lm-eval)

```bash
bash scripts/benchmark/benchmark_lm_eval.sh
```

Requires [EleutherAI/lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness). Use the model files in `src/llmtuner/model/` when loading Mistral/Llama with dropped configs.

### ⚡ LLM inference speed

```bash
bash scripts/benchmark/benchmark_speed.sh
```

Edit `model_path`, `save_file`, and `model_type` in the script before running.

<details>
<summary><strong>Quantization benchmarks (AWQ / GPTQ)</strong></summary>

```bash
bash scripts/quantization/awq.sh
bash scripts/quantization/gptq.sh
```

Edit `model_path` and `quant_path` in those scripts and ensure CUDA-compatible package versions are installed (see Installation).

</details>

## 🤖 Automated Model Selection

The model selection pipeline automates the full search over all combinations of architecture, pruning method, and drop count for a given dataset. It runs in three phases:

1. **Baseline** — fine-tunes the classification head at `drop=0` to establish a reference accuracy.
2. **Search** — quick fine-tuning (few epochs) across all `(arch × method × drop_n)` variants. Early stopping halts a search direction if accuracy drops more than a configurable threshold below baseline.
3. **Deep fine-tune** — full fine-tuning of the winning variant, followed by final test-set evaluation.

Results are logged to `results_selection/` with per-variant accuracy, the selected winner, and the final test metrics.

```bash
bash scripts/model-selection/run_selection.sh
```

Key parameters (edit `run_selection.sh`):

```bash
dataset="LCZ42"                              # Target dataset
architectures="dinov2 dinov3_vit swinv2 vit" # Architectures to search
prune_methods="block_drop layer_drop_attn layer_drop_mlp layer_drop_all"
drop_step=4              # Evaluate drop counts: 4, 8, 12, ...
early_stop_threshold=0.05 # Stop if accuracy drops >5% below baseline
baseline_epochs=5
search_epochs=5
deep_epochs=20
```

### Supported datasets

<details>
<summary><strong>Show all 10 datasets</strong></summary>

| Dataset | Domain | Classes | Notes |
|---|---|---|---|
| `imagenet-1k` | Natural images | 1000 | Standard benchmark; head fine-tuning skipped if classes match |
| `cifar10` | Natural images | 10 | |
| `LCZ42` | Remote sensing | 17 | Urban morphology classification |
| `CrossD` | Cross-domain | varies | Multi-domain classification |
| `zoolake` | Microscopy | varies | Zooplankton identification |
| `lar` | Medical | varies | Laryngeal endoscopy |
| `InfLarynge` | Medical | varies | Inflamed laryngeal tissue |
| `DAPlankton` | Microscopy | varies | Plankton imaging |
| `Bark` | Texture | varies | Tree bark classification |
| `Pest` | Agriculture | varies | Crop pest identification |

All datasets are stored as stratified HDF5 splits (`train.h5`, `val.h5`, `test.h5`) under `data/<dataset>/`. Preprocessing scripts are in `data/`.

</details>

## 🔧 Model Healing

After dropping layers, optional LoRA-based recovery fine-tuning can partially restore accuracy on the target dataset.

```bash
bash scripts/healing/heal_dropped_model.sh
```

## 📄 Citation

If you use this work, please cite the original paper:

```bibtex
@article{
    he2026uncovering,
    title={Uncovering the Redundancy in Transformers via a Unified Study of Layer Dropping},
    author={Shwai He and Guoheng Sun and Zheyu Shen and Ang Li},
    journal={Transactions on Machine Learning Research},
    issn={2835-8856},
    year={2026},
    url={https://openreview.net/forum?id=1I7PCbOPfe},
}
```

## 📬 Contact

- Alessandro Viespoli
- Original authors: Shwai He (`shwaihe@umd.edu`), Guoheng Sun (`ghsun@umd.edu`)
