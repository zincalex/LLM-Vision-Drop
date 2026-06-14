import gc
import sys
import torch

from pathlib import Path
from accelerate import Accelerator

from config import parse_args, get_drop_counts
from model_loader import (
    config_exists, setup_model_dir, swap_config,
    load_model_and_processor, cache_base_weights,
)
from evaluator import finetune_and_evaluate, deep_finetune_and_evaluate
from results_log import ResultsLog


def get_model_dir(arch):
    return f"./{arch}_model"


def validate_dataset(name, base_dir, accelerator):
    d = Path(base_dir) / name
    for s in ["train", "val", "test"]:
        if not (d / f"{s}.h5").exists():
            accelerator.print(f"ERROR: Missing {d / f'{s}.h5'}")
            sys.exit(1)


def fmt_metrics(m):
    return (f"acc={m['accuracy']:.4f}  prec={m['precision']:.4f}  "
            f"rec={m['recall']:.4f}  f1={m['f1']:.4f}")


def run_baseline(arch, method, cfg, accelerator, log, cached_weights):
    if log.has_baseline(arch, method):
        acc = log.get_baseline_accuracy(arch, method)
        accelerator.print(f"  [SKIP] Already recorded: {acc:.4f}")
        return acc

    model_dir = get_model_dir(arch)
    if not config_exists(cfg.results_prune_dir, arch, method, 0):
        accelerator.print(f"  [SKIP] No drop=0 config, skipping pair")
        return None

    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        swap_config(model_dir, cfg.results_prune_dir, arch, method, 0)
    accelerator.wait_for_everyone()
    model, processor = load_model_and_processor(model_dir, cached_state_dict=cached_weights)

    metrics = finetune_and_evaluate(
        model=model, processor=processor, accelerator=accelerator,
        dataset_name=cfg.dataset, dataset_base_dir=cfg.dataset_base_dir,
        epochs=cfg.baseline.epochs, lr=cfg.baseline.lr,
        weight_decay=cfg.baseline.weight_decay, batch_size=cfg.baseline.batch_size,
        batch_size_eval=cfg.batch_size_eval, num_workers=cfg.num_workers,
        train_split="train", eval_split="val",
    )

    del model
    torch.cuda.empty_cache()

    acc_tensor = torch.tensor([0.0], device=accelerator.device)
    if accelerator.is_main_process and metrics:
        log.record_baseline(arch, method, metrics)
        accelerator.print(f"Val: {fmt_metrics(metrics)}")
        acc_tensor[0] = metrics["accuracy"]
    if accelerator.num_processes > 1:
        torch.distributed.broadcast(acc_tensor, src=0)
    accelerator.wait_for_everyone()
    return acc_tensor.item() if acc_tensor.item() > 0 else None


def run_search(arch, method, cfg, accelerator, log, baseline_acc, cached_weights):
    drop_counts = get_drop_counts(arch, cfg.drop_step)
    model_dir = get_model_dir(arch)
    best_acc = baseline_acc

    for drop_n in drop_counts:
        label = f"{arch}/{method}/drop{drop_n}"

        if log.has_search_result(arch, method, drop_n):
            prev_acc = log.get_search_accuracy(arch, method, drop_n)
            accelerator.print(f"\n--- {label} [SKIP] already recorded: {prev_acc:.4f} ---")
            if baseline_acc is not None and (baseline_acc - prev_acc) > cfg.early_stop_threshold:
                break
            if prev_acc > best_acc:
                best_acc = prev_acc
            continue

        if not config_exists(cfg.results_prune_dir, arch, method, drop_n):
            continue

        accelerator.print(f"\n--- {label} ---")
        accelerator.wait_for_everyone()
        if accelerator.is_main_process:
            swap_config(model_dir, cfg.results_prune_dir, arch, method, drop_n)
        accelerator.wait_for_everyone()
        model, processor = load_model_and_processor(model_dir, cached_state_dict=cached_weights)

        metrics = finetune_and_evaluate(
            model=model, processor=processor, accelerator=accelerator,
            dataset_name=cfg.dataset, dataset_base_dir=cfg.dataset_base_dir,
            epochs=cfg.search.epochs, lr=cfg.search.lr,
            weight_decay=cfg.search.weight_decay, batch_size=cfg.search.batch_size,
            batch_size_eval=cfg.batch_size_eval, num_workers=cfg.num_workers,
            train_split="train", eval_split="val",
        )

        # Broadcast accuracy to all ranks so early stop is synchronized
        acc_tensor = torch.tensor([0.0], device=accelerator.device)
        if accelerator.is_main_process and metrics:
            acc_tensor[0] = metrics["accuracy"]
        if accelerator.num_processes > 1:
            torch.distributed.broadcast(acc_tensor, src=0)
        current_acc = acc_tensor.item()
        early_stopped = baseline_acc is not None and (baseline_acc - current_acc) > cfg.early_stop_threshold

        if accelerator.is_main_process and metrics:
            log.record_search_result(arch, method, drop_n, metrics, early_stopped=early_stopped)
            accelerator.print(f"Val: {fmt_metrics(metrics)}")

            if current_acc > best_acc:
                best_acc = current_acc
                accelerator.print(f"[NEW BEST] {label} — val acc: {current_acc:.4f}")

            if early_stopped:
                accelerator.print(f"[EARLY STOP] acc dropped {baseline_acc - current_acc:.4f} > {cfg.early_stop_threshold} — moving to next")

        accelerator.wait_for_everyone()
        torch.cuda.empty_cache()
        if accelerator.is_main_process:
            alloc = torch.cuda.memory_allocated() / 1024**3
            resv = torch.cuda.memory_reserved() / 1024**3
            accelerator.print(f"  GPU mem: {alloc:.1f}G allocated, {resv:.1f}G reserved")
        if early_stopped:
            break


def run_deep_finetune(winner, cfg, accelerator, log):
    arch, method, drop_n = winner["arch"], winner["method"], winner["drop_n"]

    accelerator.print(f"\n{'=' * 80}")
    accelerator.print(f"DEEP FINE-TUNING: {arch}/{method}/drop{drop_n}")
    accelerator.print(f"{'=' * 80}")

    model_dir = get_model_dir(arch)
    if accelerator.is_main_process:
        swap_config(model_dir, cfg.results_prune_dir, arch, method, drop_n)
    accelerator.wait_for_everyone()
    model, processor = load_model_and_processor(model_dir)

    test_metrics = deep_finetune_and_evaluate(
        model=model, processor=processor, accelerator=accelerator,
        dataset_name=cfg.dataset, dataset_base_dir=cfg.dataset_base_dir,
        epochs=cfg.deep_finetune.epochs, lr=cfg.deep_finetune.lr,
        weight_decay=cfg.deep_finetune.weight_decay, batch_size=cfg.deep_finetune.batch_size,
        batch_size_eval=cfg.batch_size_eval, num_workers=cfg.num_workers,
        warmup_ratio=cfg.deep_finetune.warmup_ratio,
        output_dir=cfg.output_dir,
    )

    if accelerator.is_main_process and test_metrics:
        log.record_deep_finetune(test_metrics)
        log.record_final_test(test_metrics)
        accelerator.print(f"\nTest: {fmt_metrics(test_metrics)}")

    accelerator.wait_for_everyone()


def main():
    cfg = parse_args()
    accelerator = Accelerator()

    accelerator.print("=" * 80)
    accelerator.print("AUTOMATED MODEL SELECTION PIPELINE")
    accelerator.print("=" * 80)
    accelerator.print(f"Dataset: {cfg.dataset}")
    accelerator.print(f"Architectures: {cfg.architectures}")
    accelerator.print(f"Prune methods: {cfg.prune_methods}")
    accelerator.print(f"Drop step: {cfg.drop_step}")
    accelerator.print(f"Early stop threshold: {cfg.early_stop_threshold}")
    accelerator.print(f"Num processes: {accelerator.num_processes}")
    accelerator.print("=" * 80)

    validate_dataset(cfg.dataset, cfg.dataset_base_dir, accelerator)
    log = ResultsLog(cfg.output_dir, cfg.dataset)

    for arch in cfg.architectures:
        model_dir = get_model_dir(arch)
        if not Path(model_dir).exists():
            accelerator.print(f"\n[SKIP] {model_dir} not found")
            continue

        # Copy custom .py files once per architecture
        py_copied = False
        for method in cfg.prune_methods:
            if config_exists(cfg.results_prune_dir, arch, method, 0):
                if accelerator.is_main_process:
                    setup_model_dir(model_dir, cfg.results_prune_dir, arch, method, 0)
                accelerator.wait_for_everyone()
                py_copied = True
                break
        if not py_copied:
            accelerator.print(f"\n[SKIP] No drop=0 config for {arch}")
            continue

        # Cache weights once per architecture
        if accelerator.is_main_process:
            swap_config(model_dir, cfg.results_prune_dir, arch, cfg.prune_methods[0], 0)
        accelerator.wait_for_everyone()
        cached_weights = cache_base_weights(model_dir)

        for method in cfg.prune_methods:
            accelerator.print(f"\n--- {arch} / {method} / drop0 (baseline) ---")
            baseline_acc = run_baseline(arch, method, cfg, accelerator, log, cached_weights)
            if baseline_acc is None:
                continue
            run_search(arch, method, cfg, accelerator, log, baseline_acc, cached_weights)
            torch.cuda.empty_cache()

        del cached_weights
        torch.cuda.empty_cache()

    # Force cleanup before deep finetune
    gc.collect()
    torch.cuda.empty_cache()
    accelerator.wait_for_everyone()

    # Reload log from disk so all ranks see the same results
    log = ResultsLog(cfg.output_dir, cfg.dataset)

    # Select winner
    winner = log.get_best_variant()

    if winner is None:
        accelerator.print("\nERROR: No valid variants found.")
        sys.exit(1)

    if accelerator.is_main_process:
        log.record_winner(winner["arch"], winner["method"], winner["drop_n"], winner["val_accuracy"])

    accelerator.print(f"\nWINNER: {winner['arch']}/{winner['method']}/drop{winner['drop_n']} "
                      f"(val acc: {winner['val_accuracy']:.4f})")

    # Deep fine-tune + final test
    if accelerator.is_main_process:
        setup_model_dir(
            get_model_dir(winner["arch"]), cfg.results_prune_dir,
            winner["arch"], winner["method"], winner["drop_n"]
        )
    accelerator.wait_for_everyone()
    run_deep_finetune(winner, cfg, accelerator, log)

    if accelerator.is_main_process:
        log.print_summary()

    accelerator.wait_for_everyone()


if __name__ == "__main__":
    main()
