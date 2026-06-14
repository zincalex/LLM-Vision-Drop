"""Create a val.h5 split by carving 20% out of an existing train.h5.

Usage:
    python data/create_val_from_h5.py --dataset CrossD

Reads:  data/{dataset}/train.h5
Creates:
    data/{dataset}/train.h5  (80% of original, overwritten)
    data/{dataset}/val.h5    (20% of original, new)
"""
import sys
print("Script started", flush=True)

import argparse
import h5py
import numpy as np
import random
from pathlib import Path
from collections import defaultdict

random.seed(42)
np.random.seed(42)


def stratified_split(labels, val_ratio=0.2):
    by_label = defaultdict(list)
    for idx, label in enumerate(labels):
        by_label[label].append(idx)
    train_idx, val_idx = [], []
    for label, indices in by_label.items():
        random.shuffle(indices)
        n_val = max(1, int(len(indices) * val_ratio))
        val_idx.extend(indices[:n_val])
        train_idx.extend(indices[n_val:])
    return sorted(train_idx), sorted(val_idx)


def save_split(images, labels, indices, output_path, desc=""):
    imgs = images[indices]
    lbls = labels[indices]
    print(f"  {desc}: {len(indices)} samples, shape={imgs.shape}", flush=True)
    with h5py.File(output_path, "w") as f:
        f.create_dataset("images", data=imgs, compression="gzip", compression_opts=4)
        f.create_dataset("labels", data=lbls, compression="gzip", compression_opts=4)
        f.attrs["num_samples"] = len(indices)
        f.attrs["num_classes"] = len(np.unique(lbls))
        f.attrs["image_shape"] = imgs.shape[1:]
    size_mb = output_path.stat().st_size / 1024**2
    print(f"  Saved -> {output_path} ({size_mb:.1f} MB)", flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--data_dir", type=str, default="data")
    parser.add_argument("--val_ratio", type=float, default=0.2)
    args = parser.parse_args()

    dataset_dir = Path(args.data_dir) / args.dataset
    train_path = dataset_dir / "train.h5"

    print(f"Dataset: {args.dataset}", flush=True)
    print(f"Reading: {train_path}", flush=True)

    with h5py.File(train_path, "r") as f:
        images = f["images"][:]
        labels = f["labels"][:]

    print(f"Total samples: {len(images)}", flush=True)
    print(f"Classes: {len(np.unique(labels))}", flush=True)

    train_idx, val_idx = stratified_split(labels, args.val_ratio)
    print(f"Split: train={len(train_idx)}, val={len(val_idx)}", flush=True)

    # Save val first, then overwrite train
    save_split(images, labels, val_idx, dataset_dir / "val.h5", desc="Val")
    save_split(images, labels, train_idx, train_path, desc="Train")

    print("Done!", flush=True)


if __name__ == "__main__":
    main()
