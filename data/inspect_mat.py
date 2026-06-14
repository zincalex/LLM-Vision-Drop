"""Inspect a .mat file: print structure, dataset info, and save sample images.

Usage:
    python data/inspect_mat.py myfile.mat
"""
import os
import argparse
import numpy as np
from collections import defaultdict


def load_mat(path):
    try:
        import scipy.io as sio
        data = sio.loadmat(path)
        return data, "v5 (scipy)"
    except NotImplementedError:
        pass
    import h5py
    print("Loading as MATLAB v7.3 (HDF5)...\n")
    f = h5py.File(path, "r")
    return f, "v7.3 (HDF5)"


def describe_cell(val, depth=0, max_children=10):
    """Recursively describe a cell/object array or HDF5 group."""
    import h5py
    prefix = "  " * (depth + 1)

    if isinstance(val, h5py.Group):
        n = len(val)
        print(f"{prefix}Group ({n} items)")
        keys = list(val.keys())
        for k in keys[:max_children]:
            child = val[k]
            if isinstance(child, h5py.Dataset):
                shape = child.shape
                print(f"{prefix}  {k}: Dataset shape={shape}, dtype={child.dtype}")
            elif isinstance(child, h5py.Group):
                print(f"{prefix}  {k}: Group ({len(child)} items)")
            else:
                print(f"{prefix}  {k}: {type(child).__name__}")
        if n > max_children:
            print(f"{prefix}  ... ({n - max_children} more items)")
        return

    if isinstance(val, h5py.Dataset):
        shape = val.shape
        dtype = val.dtype
        extra = ""
        if val.size > 0 and val.size < 1e7 and np.issubdtype(dtype, np.number):
            arr = val[:]
            extra = f", min={arr.min()}, max={arr.max()}, unique={len(np.unique(arr))}"
        print(f"{prefix}Dataset shape={shape}, dtype={dtype}{extra}")
        return

    if not isinstance(val, np.ndarray):
        print(f"{prefix}type={type(val).__name__}, value={val}")
        return

    if val.dtype == object:
        print(f"{prefix}Cell array, shape={val.shape}")
        for idx in np.ndindex(val.shape):
            inner = val[idx]
            label = f"[{','.join(str(i) for i in idx)}]"
            if isinstance(inner, np.ndarray):
                if inner.dtype == object:
                    print(f"{prefix}  {label}: Cell array, shape={inner.shape}")
                    # Show first few elements
                    flat = inner.flatten()
                    for j, item in enumerate(flat[:3]):
                        if isinstance(item, np.ndarray):
                            print(f"{prefix}    [{j}]: ndarray shape={item.shape}, dtype={item.dtype}")
                        else:
                            print(f"{prefix}    [{j}]: {type(item).__name__} = {item}")
                    if len(flat) > 3:
                        print(f"{prefix}    ... ({len(flat)} items total)")
                else:
                    extra = ""
                    if np.issubdtype(inner.dtype, np.number) and inner.size > 0:
                        extra = f", min={inner.min()}, max={inner.max()}, unique={len(np.unique(inner))}"
                    print(f"{prefix}  {label}: ndarray shape={inner.shape}, dtype={inner.dtype}{extra}")
            else:
                print(f"{prefix}  {label}: {type(inner).__name__} = {inner}")
    else:
        extra = ""
        if np.issubdtype(val.dtype, np.number) and val.size > 0:
            extra = f", min={val.min()}, max={val.max()}, unique={len(np.unique(val))}"
        print(f"{prefix}ndarray shape={val.shape}, dtype={val.dtype}{extra}")


def try_parse_dataset(data):
    """Try to extract images, labels, and other info from common .mat structures."""
    print(f"\n{'='*60}")
    print("DATASET ANALYSIS")
    print(f"{'='*60}")

    # Check for DATA cell array (common format)
    if "DATA" in data and isinstance(data["DATA"], np.ndarray) and data["DATA"].dtype == object:
        cell = data["DATA"]
        print(f"\nDATA cell array: shape={cell.shape}")

        # Try to find images (object array of variable-size arrays)
        images, labels = None, None
        for i in range(cell.shape[1]):
            val = cell[0, i]
            if not isinstance(val, np.ndarray):
                continue

            if val.dtype == object:
                # Likely images
                flat = val.flatten()
                if len(flat) > 0 and isinstance(flat[0], np.ndarray) and flat[0].ndim >= 2:
                    images = flat
                    print(f"\n  Images found at DATA[0,{i}]:")
                    print(f"    Count: {len(images)}")
                    shapes = [img.shape for img in images]
                    heights = [s[0] for s in shapes]
                    widths = [s[1] for s in shapes]
                    channels = set(s[2] for s in shapes if len(s) == 3)
                    print(f"    Height range: [{min(heights)}, {max(heights)}]")
                    print(f"    Width range: [{min(widths)}, {max(widths)}]")
                    print(f"    Channels: {channels if channels else 'grayscale'}")
                    uniform = len(set(shapes)) == 1
                    print(f"    Uniform size: {uniform}" + (f" ({shapes[0]})" if uniform else ""))

            elif val.ndim <= 2 and np.issubdtype(val.dtype, np.integer):
                flat = val.flatten()
                unique = np.unique(flat)
                # Heuristic: labels have repeated values, fewer unique than total, and fewer classes than PERM
                if len(unique) < len(flat) and len(unique) <= 1000:
                    # Prefer the array with fewer unique values (real labels vs PERM indices)
                    if labels is None or len(unique) < len(np.unique(labels)):
                        labels = flat
                        print(f"\n  Labels found at DATA[0,{i}]:")
                        print(f"    Count: {len(labels)}")
                        print(f"    Num classes: {len(unique)}")
                        print(f"    Label values: {unique}")
                        counts = defaultdict(int)
                        for l in labels:
                            counts[int(l)] += 1
                        print(f"    Per-class distribution:")
                        for l in sorted(counts.keys()):
                            print(f"      Label {l}: {counts[l]} samples")
                    else:
                        print(f"\n  DATA[0,{i}]: numeric array shape={val.shape}, "
                              f"unique={len(unique)} (skipped, likely PERM/fold indices)")
                else:
                    print(f"\n  DATA[0,{i}]: numeric array shape={val.shape}, "
                          f"unique={len(unique)}, range=[{val.min()}, {val.max()}]")
                    if val.size == 1:
                        print(f"    Scalar value: {val.flatten()[0]}")

        if images is not None and labels is not None and len(images) == len(labels):
            print(f"\n  Summary: {len(images)} images, {len(np.unique(labels))} classes")
        elif images is not None:
            print(f"\n  Summary: {len(images)} images (labels not identified)")

    # Check for PERM (cross-validation permutations)
    if "PERM" in data:
        perm = data["PERM"]
        if isinstance(perm, np.ndarray) and perm.ndim == 2:
            n_perms, n_samples_perm = perm.shape
            print(f"\n  PERM: shape={perm.shape}")
            print(f"    {n_perms} permutations, {n_samples_perm} samples each")
            print(f"    Values range: [{int(perm.min())}, {int(perm.max())}] (1-indexed)")

            # Try to get train size from DATA[0,3]
            train_sizes = None
            if "DATA" in data and isinstance(data["DATA"], np.ndarray) and data["DATA"].dtype == object:
                cell = data["DATA"]
                if cell.shape[1] >= 4:
                    hint = cell[0, 3].flatten()
                    if len(hint) == 1:
                        # Single value — same train size for all permutations
                        train_sizes = np.array([int(hint[0])] * n_perms)
                    elif len(hint) == n_perms:
                        train_sizes = hint.astype(int)

            if train_sizes is not None:
                print(f"\n    Train sizes per permutation: {train_sizes.tolist()}")
                train_size = train_sizes[0]
                test_size = n_samples_perm - train_size
                perm_row = perm[0].flatten().astype(int) - 1  # 0-indexed
                train_idx = perm_row[:train_size]
                test_idx = perm_row[train_size:]
                train_labels = labels[train_idx]
                test_labels = labels[test_idx]
                print(f"\n    Round 1 (PERM row 0, train_size={train_size}):")
                print(f"      Train: {len(train_idx)} samples, {len(np.unique(train_labels))} classes")
                print(f"      Test: {len(test_idx)} samples, {len(np.unique(test_labels))} classes")
                missing_test = set(np.unique(labels)) - set(np.unique(test_labels))
                missing_train = set(np.unique(labels)) - set(np.unique(train_labels))
                if missing_test:
                    print(f"      ⚠ Test missing classes: {sorted(int(x) for x in missing_test)}")
                if missing_train:
                    print(f"      ⚠ Train missing classes: {sorted(int(x) for x in missing_train)}")
                test_counts = defaultdict(int)
                for l in test_labels:
                    test_counts[int(l)] += 1
                print(f"      Test per-class:")
                for lbl in sorted(test_counts.keys()):
                    print(f"        Label {lbl}: {test_counts[lbl]}")

                # Analyze index grouping (patient/tree-level splits)
                print(f"\n      Index grouping analysis:")
                train_sorted = sorted(train_idx)
                test_sorted = sorted(test_idx)
                # Count contiguous runs in test
                test_runs = 1
                for i in range(1, len(test_sorted)):
                    if test_sorted[i] != test_sorted[i-1] + 1:
                        test_runs += 1
                train_runs = 1
                for i in range(1, len(train_sorted)):
                    if train_sorted[i] != train_sorted[i-1] + 1:
                        train_runs += 1
                print(f"        Train: {len(train_sorted)} samples in {train_runs} contiguous blocks")
                print(f"        Test: {len(test_sorted)} samples in {test_runs} contiguous blocks")
                if test_runs < 20:
                    print(f"        Test blocks: ", end="")
                    blocks = []
                    start = test_sorted[0]
                    for i in range(1, len(test_sorted)):
                        if test_sorted[i] != test_sorted[i-1] + 1:
                            blocks.append(f"[{start}-{test_sorted[i-1]}]({test_sorted[i-1]-start+1})")
                            start = test_sorted[i]
                    blocks.append(f"[{start}-{test_sorted[-1]}]({test_sorted[-1]-start+1})")
                    print(", ".join(blocks))
            else:
                # Fallback: try different fold counts
                perm_row = perm[0].flatten().astype(int) - 1
                print(f"\n    Round 1 analysis (PERM row 0):")
                for k in [3, 5, 10]:
                    fold_size = n_samples_perm // k
                    if fold_size == 0:
                        continue
                    test_idx = perm_row[:fold_size]
                    train_idx = perm_row[fold_size:]
                    test_labels_k = labels[test_idx]
                    train_labels_k = labels[train_idx]
                    print(f"\n      If {k}-fold CV:")
                    print(f"        Test: {fold_size} samples, {len(np.unique(test_labels_k))} classes")
                    print(f"        Train: {n_samples_perm - fold_size} samples, {len(np.unique(train_labels_k))} classes")

    return images, labels


def try_parse_dataset_v73(f):
    """Parse a v7.3 HDF5 .mat file with DATA cell array."""
    import h5py
    print(f"\n{'='*60}")
    print("DATASET ANALYSIS (v7.3)")
    print(f"{'='*60}")

    if "DATA" not in f:
        print("No DATA key found.")
        return None, None

    data_ds = f["DATA"]
    # DATA is (5,1) of object references — each ref points into #refs#
    refs = data_ds[:].flatten()  # 5 object references
    print(f"\nDATA: {len(refs)} cells")

    images, labels = None, None

    for i, ref in enumerate(refs):
        obj = f[ref]
        if isinstance(obj, h5py.Dataset):
            arr = obj[:]
            if arr.ndim == 2 and arr.dtype == object:
                # Cell array of object references (images)
                print(f"\n  Images found at cell {i} (ref array):")
                img_refs = arr.flatten()
                n = len(img_refs)
                print(f"    Count: {n}")
                # Sample a few shapes
                shapes = []
                for r in img_refs[:50]:
                    try:
                        ds = f[r]
                        if isinstance(ds, h5py.Dataset):
                            shapes.append(ds.shape)
                    except Exception:
                        pass
                if shapes:
                    heights = [s[1] if s[0] in (1,3) else s[0] for s in shapes]
                    widths = [s[2] if s[0] in (1,3) else s[1] for s in shapes]
                    print(f"    Format: channels-first (C, H, W)")
                    print(f"    Height range: [{min(heights)}, {max(heights)}]")
                    print(f"    Width range: [{min(widths)}, {max(widths)}]")
                    uniform = len(set(shapes)) == 1
                    print(f"    Uniform size: {uniform}" + (f" ({shapes[0]})" if uniform else ""))
                # Load all images
                print(f"    Loading images...")
                img_list = []
                for r in img_refs:
                    try:
                        ds = f[r]
                        if isinstance(ds, h5py.Dataset):
                            # MATLAB HDF5: (C, W, H) → transpose to (H, W, C)
                            img_list.append(np.transpose(ds[:], (2, 1, 0)))
                    except Exception:
                        pass
                images = img_list
                print(f"    Loaded {len(images)} images")
                continue

            if arr.ndim <= 2 and np.issubdtype(arr.dtype, np.number):
                flat = arr.flatten()
                unique = np.unique(flat)
                if len(unique) < len(flat) and len(unique) <= 1000:
                    if labels is None or len(unique) < len(np.unique(labels)):
                        labels = flat
                        print(f"\n  Labels found at cell {i}:")
                        print(f"    Count: {len(labels)}")
                        print(f"    Num classes: {len(unique)}")
                        print(f"    Label values: {unique}")
                        counts = defaultdict(int)
                        for l in labels:
                            counts[int(l)] += 1
                        print(f"    Per-class distribution:")
                        for l in sorted(counts.keys()):
                            print(f"      Label {l}: {counts[l]} samples")
                    else:
                        print(f"\n  Cell {i}: numeric shape={arr.shape}, unique={len(unique)} (skipped, likely PERM)")
                elif arr.size == 1:
                    print(f"\n  Cell {i}: scalar = {flat[0]}")
                else:
                    print(f"\n  Cell {i}: numeric shape={arr.shape}, unique={len(unique)}, range=[{arr.min()}, {arr.max()}]")

        elif isinstance(obj, h5py.Group):
            # Group of object references = cell array of images
            keys = list(obj.keys())
            n = len(keys)
            # Check if these are image datasets
            sample = obj[keys[0]]
            if isinstance(sample, h5py.Dataset) and sample.ndim == 3:
                print(f"\n  Images found at cell {i}:")
                print(f"    Count: {n}")
                # Sample a few to get size range
                sample_keys = keys[:50] + keys[-50:] if n > 100 else keys
                shapes = []
                for k in sample_keys:
                    ds = obj[k]
                    if isinstance(ds, h5py.Dataset):
                        shapes.append(ds.shape)
                if shapes:
                    # Channels-first: (C, H, W)
                    heights = [s[1] for s in shapes]
                    widths = [s[2] for s in shapes]
                    channels = set(s[0] for s in shapes)
                    print(f"    Format: channels-first (C, H, W)")
                    print(f"    Height range: [{min(heights)}, {max(heights)}]")
                    print(f"    Width range: [{min(widths)}, {max(widths)}]")
                    print(f"    Channels: {channels}")
                    uniform = len(set(shapes)) == 1
                    print(f"    Uniform size: {uniform}" + (f" ({shapes[0]})" if uniform else ""))

                # Load images as list of arrays (channels-first → channels-last)
                print(f"    Loading images...")
                img_list = []
                for k in keys:
                    ds = obj[k]
                    if isinstance(ds, h5py.Dataset):
                        arr = ds[:].transpose(2, 1, 0)  # (C,H,W) → (W,H,C) → need (H,W,C)
                        # HDF5 MATLAB stores as (C, W, H) transposed, so: (C,col,row) → (row,col,C)
                        arr = np.transpose(ds[:], (2, 1, 0))
                        img_list.append(arr)
                images = img_list
                print(f"    Loaded {len(images)} images")
            else:
                print(f"\n  Cell {i}: Group with {n} items (not images)")

    if images is not None and labels is not None:
        print(f"\n  Summary: {len(images)} images, {len(np.unique(labels))} classes")

        # Analyze PERM from cell 2 if available
        perm_arr = None
        train_size_hint = None
        for i, ref in enumerate(refs):
            obj = f[ref]
            if isinstance(obj, h5py.Dataset):
                arr = obj[:]
                flat = arr.flatten()
                if arr.ndim == 2 and np.issubdtype(arr.dtype, np.number):
                    unique = np.unique(flat)
                    n = len(images)
                    # PERM: unique count matches sample count
                    if len(unique) == n and perm_arr is None:
                        perm_arr = arr
                        if perm_arr.shape[0] > perm_arr.shape[1]:
                            perm_arr = perm_arr.T
                    # Train size hint: small array with few values
                    elif arr.size <= 10 and arr.size > 1:
                        train_size_hint = arr.flatten()

        if perm_arr is not None:
            n_perms, n_samples_perm = perm_arr.shape
            print(f"\n  PERM: shape={perm_arr.shape}")
            print(f"    {n_perms} permutations, {n_samples_perm} samples each")
            print(f"    Values range: [{int(perm_arr.min())}, {int(perm_arr.max())}]")

            # Use DATA[0,3] as train sizes per permutation
            train_sizes = None
            for i, ref in enumerate(refs):
                obj = f[ref]
                if isinstance(obj, h5py.Dataset):
                    arr = obj[:]
                    if arr.ndim <= 2 and np.issubdtype(arr.dtype, np.number):
                        vals = arr.flatten().astype(int)
                        # Match: array with n_perms values, all valid train sizes
                        if len(vals) == n_perms and all(0 < v < n_samples_perm for v in vals):
                            train_sizes = vals
                        # Match: scalar, use as train size for all permutations
                        elif len(vals) == 1 and 0 < vals[0] < n_samples_perm and train_sizes is None:
                            train_sizes = np.array([int(vals[0])] * n_perms)

            if train_sizes is not None:
                print(f"\n    Train sizes per permutation: {train_sizes.tolist()}")
                for p_idx in range(min(1, n_perms)):  # Just show round 1
                    train_size = train_sizes[p_idx]
                    test_size = n_samples_perm - train_size
                    perm_row = perm_arr[p_idx].flatten().astype(int) - 1  # 0-indexed
                    labels_arr = np.array([int(l) for l in labels])
                    train_idx = perm_row[:train_size]
                    test_idx = perm_row[train_size:]
                    train_idx = train_idx[(train_idx >= 0) & (train_idx < len(labels_arr))]
                    test_idx = test_idx[(test_idx >= 0) & (test_idx < len(labels_arr))]
                    train_labels = labels_arr[train_idx]
                    test_labels = labels_arr[test_idx]
                    print(f"\n    Round {p_idx + 1} (PERM row {p_idx}, train_size={train_size}):")
                    print(f"      Train: {len(train_idx)} samples, {len(np.unique(train_labels))} classes")
                    print(f"      Test: {len(test_idx)} samples, {len(np.unique(test_labels))} classes")
                    missing_test = set(np.unique(labels_arr)) - set(np.unique(test_labels))
                    missing_train = set(np.unique(labels_arr)) - set(np.unique(train_labels))
                    if missing_test:
                        print(f"      ⚠ Test missing classes: {sorted(int(x) for x in missing_test)}")
                    if missing_train:
                        print(f"      ⚠ Train missing classes: {sorted(int(x) for x in missing_train)}")
                    # Show per-class distribution for test
                    test_counts = defaultdict(int)
                    for l in test_labels:
                        test_counts[int(l)] += 1
                    print(f"      Test per-class:")
                    for lbl in sorted(test_counts.keys()):
                        print(f"        Label {lbl}: {test_counts[lbl]}")

                    # Analyze index grouping
                    print(f"\n      Index grouping analysis:")
                    train_sorted = sorted(train_idx)
                    test_sorted = sorted(test_idx)
                    test_runs = 1
                    for i in range(1, len(test_sorted)):
                        if test_sorted[i] != test_sorted[i-1] + 1:
                            test_runs += 1
                    train_runs = 1
                    for i in range(1, len(train_sorted)):
                        if train_sorted[i] != train_sorted[i-1] + 1:
                            train_runs += 1
                    print(f"        Train: {len(train_sorted)} samples in {train_runs} contiguous blocks")
                    print(f"        Test: {len(test_sorted)} samples in {test_runs} contiguous blocks")
                    if test_runs < 20:
                        print(f"        Test blocks: ", end="")
                        blocks = []
                        start = test_sorted[0]
                        for i in range(1, len(test_sorted)):
                            if test_sorted[i] != test_sorted[i-1] + 1:
                                blocks.append(f"[{start}-{test_sorted[i-1]}]({test_sorted[i-1]-start+1})")
                                start = test_sorted[i]
                        blocks.append(f"[{start}-{test_sorted[-1]}]({test_sorted[-1]-start+1})")
                        print(", ".join(blocks))
            else:
                # Fallback: try different fold counts
                perm_row = perm_arr[0].flatten().astype(int) - 1
                labels_arr = np.array([int(l) for l in labels])
                print(f"\n    Round 1 analysis (PERM row 0):")
                for k in [3, 5, 10]:
                    fold_size = n_samples_perm // k
                    if fold_size == 0:
                        continue
                    test_idx = perm_row[:fold_size]
                    train_idx = perm_row[fold_size:]
                    test_idx = test_idx[(test_idx >= 0) & (test_idx < len(labels_arr))]
                    train_idx = train_idx[(train_idx >= 0) & (train_idx < len(labels_arr))]
                    test_labels = labels_arr[test_idx]
                    train_labels = labels_arr[train_idx]
                    print(f"\n      If {k}-fold CV:")
                    print(f"        Test: {fold_size} samples, {len(np.unique(test_labels))} classes")
                    print(f"        Train: {n_samples_perm - fold_size} samples, {len(np.unique(train_labels))} classes")

    elif images is not None:
        print(f"\n  Summary: {len(images)} images (labels not identified)")

    return images, labels


def save_samples(images, labels, output_dir, n=5):
    """Save n sample images from different labels as .jpg files."""
    from PIL import Image
    os.makedirs(output_dir, exist_ok=True)

    def to_pil(img):
        """Convert any image array layout to PIL Image."""
        if img.ndim == 2:
            return Image.fromarray(img.astype(np.uint8), mode="L")
        # If channels-first (C, H, W) with C in {1, 3}
        if img.ndim == 3 and img.shape[0] in (1, 3):
            img = np.transpose(img, (1, 2, 0))
        if img.ndim == 3 and img.shape[2] == 1:
            img = img.squeeze(-1)
            return Image.fromarray(img.astype(np.uint8), mode="L")
        return Image.fromarray(img.astype(np.uint8))

    if labels is not None:
        seen = set()
        for idx, (img, lbl) in enumerate(zip(images, labels)):
            lbl = int(lbl)
            if lbl in seen:
                continue
            seen.add(lbl)
            pil = to_pil(img)
            out = os.path.join(output_dir, f"label_{lbl}_idx_{idx}.jpg")
            pil.save(out)
            print(f"  Saved: {out} (label={lbl}, size={img.shape})")
            if len(seen) >= n:
                break
    else:
        for idx in range(min(n, len(images))):
            img = images[idx]
            pil = to_pil(img)
            out = os.path.join(output_dir, f"idx_{idx}.jpg")
            pil.save(out)
            print(f"  Saved: {out} (size={img.shape})")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("path", type=str, help="Name of .mat file")
    parser.add_argument("--output_dir", type=str, default="mat_samples")
    parser.add_argument("--no_save", action="store_true", help="Skip saving sample images")
    args = parser.parse_args()

    path = args.path if os.path.sep in args.path else os.path.join(os.getcwd(), args.path)
    print(f"File: {path}\n")

    data, fmt = load_mat(path)
    print(f"Format: {fmt}")
    keys = [k for k in data.keys() if not k.startswith("__")]
    print(f"Top-level keys: {keys}")

    print(f"\n{'='*60}")
    print("RAW STRUCTURE")
    print(f"{'='*60}")
    for key in keys:
        print(f"\n--- {key} ---")
        describe_cell(data[key])

    # Only run dataset analysis for v5 (dict) format
    images, labels = None, None
    if isinstance(data, dict):
        images, labels = try_parse_dataset(data)
    else:
        images, labels = try_parse_dataset_v73(data)

    if not args.no_save and images is not None:
        print(f"\n{'='*60}")
        print("SAMPLE IMAGES")
        print(f"{'='*60}")
        save_samples(images, labels, args.output_dir)
        print(f"\nSamples saved to {args.output_dir}/")

    # Close h5py file if needed
    import h5py
    if isinstance(data, h5py.File):
        data.close()


if __name__ == "__main__":
    main()
