"""Process ImageNet-1k dataset into train/val/test H5 splits.

Split strategy:
    - Loads ImageNet validation split from HuggingFace (streaming)
    - Deterministic split: idx%5 < 3 = train pool, rest = test (60/40)
    - Train pool further split 80/20 stratified per class into train/val
    - Images resized to most common size and stored as fixed-size H5 arrays
"""
import argparse
import h5py
import json
import random
import warnings
import numpy as np
from pathlib import Path
from collections import defaultdict
from datetime import datetime
from PIL import Image
from tqdm import tqdm

random.seed(42)
np.random.seed(42)
warnings.filterwarnings("ignore", category=UserWarning, module="PIL")


def stratified_split(images, labels, val_ratio=0.2):
    by_label = defaultdict(list)
    for i, lbl in enumerate(labels):
        by_label[lbl].append(i)
    train_idx, val_idx = [], []
    for lbl in sorted(by_label.keys()):
        idxs = by_label[lbl]
        random.shuffle(idxs)
        n_val = max(1, int(len(idxs) * val_ratio))
        val_idx.extend(idxs[:n_val])
        train_idx.extend(idxs[n_val:])
    random.shuffle(train_idx)
    random.shuffle(val_idx)
    return train_idx, val_idx


def save_h5_fixed(path, images, labels, desc=""):
    sizes = defaultdict(int)
    for img in images:
        sizes[img.size] += 1
    target_size = max(sizes, key=sizes.get)
    if len(sizes) > 1:
        print(f"  Found {len(sizes)} sizes, resizing to {target_size}")

    images_np = []
    for img in images:
        if img.size != target_size:
            img = img.resize(target_size, Image.LANCZOS)
        images_np.append(np.array(img, dtype=np.uint8))

    images_arr = np.array(images_np, dtype=np.uint8)
    labels_arr = np.array(labels, dtype=np.int32)

    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(path, "w") as f:
        f.create_dataset("images", data=images_arr, compression="gzip", compression_opts=4)
        f.create_dataset("labels", data=labels_arr, compression="gzip", compression_opts=4)
        f.attrs["num_samples"] = len(images_np)
        f.attrs["num_classes"] = len(np.unique(labels_arr))
        f.attrs["image_shape"] = images_arr.shape[1:]
    size_mb = Path(path).stat().st_size / 1024**2
    print(f"  {desc}: {len(images_np)} samples, shape={images_arr.shape} ({size_mb:.1f} MB)")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", type=str, default="data/imagenet-1k")
    parser.add_argument("--max_samples", type=int, default=None)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    print(f"{'='*60}")
    print(f"Processing ImageNet-1k dataset")
    print(f"{'='*60}")

    from datasets import load_dataset
    print("Loading ImageNet validation split (streaming)...")
    val_dataset = load_dataset("ILSVRC/imagenet-1k", split="validation", streaming=True)

    train_pool_images, train_pool_labels = [], []
    test_images, test_labels = [], []

    print("Processing images...")
    for idx, sample in enumerate(val_dataset):
        if args.max_samples and idx >= args.max_samples:
            break
        if idx % 5000 == 0 and idx > 0:
            print(f"  Processed {idx} images...")

        img = sample["image"]
        if img.mode != "RGB":
            img = img.convert("RGB")
        label = sample["label"]

        if (idx % 5) < 3:
            train_pool_images.append(img)
            train_pool_labels.append(label)
        else:
            test_images.append(img)
            test_labels.append(label)

    print(f"\nTrain pool: {len(train_pool_images)}, Test: {len(test_images)}")

    train_idx, val_idx = stratified_split(train_pool_images, train_pool_labels, val_ratio=0.2)
    train_images = [train_pool_images[i] for i in train_idx]
    train_labels = [train_pool_labels[i] for i in train_idx]
    val_images = [train_pool_images[i] for i in val_idx]
    val_labels = [train_pool_labels[i] for i in val_idx]
    print(f"Train: {len(train_images)}, Val: {len(val_images)}, Test: {len(test_images)}")

    print(f"\nSaving H5 files...")
    save_h5_fixed(output_dir / "train.h5", train_images, train_labels, desc="Train")
    save_h5_fixed(output_dir / "val.h5", val_images, val_labels, desc="Val")
    save_h5_fixed(output_dir / "test.h5", test_images, test_labels, desc="Test")

    metadata = {
        "dataset_name": "imagenet-1k",
        "num_classes": 1000,
        "total_samples": len(train_pool_images) + len(test_images),
        "split_policy": "validation split: idx%5<3 = train pool, rest = test; train pool 80/20 stratified into train/val",
        "seed": 42,
        "labels_0_indexed": True,
        "variable_size_images": False,
        "created_at": datetime.now().isoformat(),
        "splits": {
            "train": {"num_samples": len(train_images)},
            "val": {"num_samples": len(val_images)},
            "test": {"num_samples": len(test_images)},
        },
    }
    with open(output_dir / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"\nMetadata saved to {output_dir / 'metadata.json'}")

    print(f"\n{'='*60}")
    print(f"DONE: train={len(train_images)}, val={len(val_images)}, test={len(test_images)}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
