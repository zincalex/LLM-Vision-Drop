import re
import csv
import argparse

from pathlib import Path

# SDR (Speedup Degradation Ratio): γ = ΔAvg / ΔSpeedup
# Measures performance degradation per 1% speedup increase
# Lower γ = better efficiency (less accuracy loss per speedup gain)


def parse_accuracy_from_file(filepath: Path) -> float:
    try:
        with open(filepath, 'r') as f:
            for line in f:
                if 'Accuracy:' in line:
                    match = re.search(r'Accuracy:\s+([\d.]+)', line)
                    if match:
                        return float(match.group(1)) * 100
    except Exception:
        pass

    return None


def parse_throughput_from_csv(filepath: Path, batch_size=128) -> float:
    try:
        with open(filepath, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                if int(row['Batch Size']) == batch_size:
                    return float(row['Throughput (imgs/s)'])
    except Exception:
        pass

    return None


def collect_accuracy_data(model, prune_method, drop_num, datasets, results_dir):
    accuracies = {}
    for dataset in datasets:
        filename = f"output_vision_{model}_drop{drop_num}_{prune_method}_{dataset}.out"
        filepath = Path(results_dir) / dataset / filename
        
        if filepath.exists():
            accuracy = parse_accuracy_from_file(filepath)
            if accuracy is not None:
                accuracies[dataset] = accuracy
    
    return accuracies


def get_baseline_accuracy(model, prune_method, datasets, results_dir):
    return collect_accuracy_data(model, prune_method, 0, datasets, results_dir)


def get_baseline_throughput(model, speed_dir):
    filepath = Path(speed_dir) / f"{model}_original_speed.csv"
    if filepath.exists():
        throughput = parse_throughput_from_csv(filepath, batch_size=128)
        if throughput is not None:
            return throughput
    
    return None


def get_dropped_throughput(model, prune_method, drop_num, speed_dir):
    filepath = Path(speed_dir) / f"{model}_drop{drop_num}_{prune_method}_speed.csv"
    if filepath.exists():
        throughput = parse_throughput_from_csv(filepath, batch_size=128)
        if throughput is not None:
            return throughput
    
    return None


def calculate_sdr(baseline_acc_dict, dropped_acc_dict, baseline_throughput, dropped_throughput):
    common_datasets = set(baseline_acc_dict.keys()) & set(dropped_acc_dict.keys())
    
    if not common_datasets:
        return None, None, None, None, None
    
    baseline_avg = sum(baseline_acc_dict[d] for d in common_datasets) / len(common_datasets)
    dropped_avg = sum(dropped_acc_dict[d] for d in common_datasets) / len(common_datasets)
    
    delta_avg = ((baseline_avg - dropped_avg) / baseline_avg) * 100
    delta_speedup = ((dropped_throughput - baseline_throughput) / baseline_throughput) * 100
    
    if delta_speedup == 0:
        sdr = None
    else:
        sdr = delta_avg / delta_speedup
    
    return baseline_avg, dropped_avg, delta_avg, delta_speedup, sdr


def write_sdr_report(output_path, drop_num, baseline_acc_dict, dropped_acc_dict, baseline_throughput, dropped_throughput,
                     baseline_avg, dropped_avg, delta_avg, delta_speedup, sdr):
    
    with open(output_path, 'w') as f:
        f.write("-" * 80 + "\n")
        f.write("BASELINE (drop_num=0)\n")
        f.write("-" * 80 + "\n")
        f.write(f"Throughput: {baseline_throughput:.2f} imgs/s\n")
        f.write(f"Average Accuracy: {baseline_avg:.2f}%\n\n")
        f.write("Per-Dataset Accuracy:\n")
        for dataset in sorted(baseline_acc_dict.keys()):
            f.write(f"  - {dataset}: {baseline_acc_dict[dataset]:.2f}%\n")
        f.write("\n")
        
        f.write("-" * 80 + "\n")
        f.write(f"DROPPED MODEL (drop_num={drop_num})\n")
        f.write("-" * 80 + "\n")
        f.write(f"Throughput: {dropped_throughput:.2f} imgs/s\n")
        f.write(f"Average Accuracy: {dropped_avg:.2f}%\n\n")
        f.write("Per-Dataset Accuracy:\n")
        for dataset in sorted(dropped_acc_dict.keys()):
            f.write(f"  - {dataset}: {dropped_acc_dict[dataset]:.2f}%\n")
        f.write("\n")
        
        f.write("-" * 80 + "\n")
        f.write("METRICS\n")
        f.write("-" * 80 + "\n")
        f.write(f"ΔAvg (Accuracy Loss):     {delta_avg:+.2f}%\n")
        f.write(f"ΔSpeedup (Throughput Gain): {delta_speedup:+.2f}%\n")
        
        if sdr is not None:
            f.write(f"SDR (γ):                   {sdr:.4f}\n\n")
            
            f.write("-" * 80 + "\n")
            f.write("INTERPRETATION\n")
            f.write("-" * 80 + "\n")
            f.write(f"For every 1% increase in speedup, accuracy drops by {abs(sdr):.4f}%\n")
        else:
            f.write(f"SDR (γ):                   N/A (zero speedup)\n")


def compute_sdr_for_configuration(model, prune_method, drop_num, datasets, 
                                   results_dir, speed_dir, output_dir):
    
    print(f"Processing: {model} - {prune_method} - drop{drop_num}")
    
    baseline_acc_dict = get_baseline_accuracy(model, prune_method, datasets, results_dir)
    baseline_throughput = get_baseline_throughput(model, speed_dir)
    dropped_acc_dict = collect_accuracy_data(model, prune_method, drop_num, datasets, results_dir)
    dropped_throughput = get_dropped_throughput(model, prune_method, drop_num, speed_dir)
    baseline_avg, dropped_avg, delta_avg, delta_speedup, sdr = calculate_sdr(
        baseline_acc_dict, dropped_acc_dict, baseline_throughput, dropped_throughput
    )

    output_path = Path(output_dir) / f"SDR_{model}_{prune_method}_drop{drop_num}.txt"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    write_sdr_report(output_path, drop_num, baseline_acc_dict, dropped_acc_dict, baseline_throughput, dropped_throughput,
        baseline_avg, dropped_avg, delta_avg, delta_speedup, sdr)
    print(f"✓ SDR computation completed successfully")

    return True


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--prune_method", type=str, required=True)
    parser.add_argument("--drop_num", type=int, required=True)
    parser.add_argument("--datasets", nargs='+', default=["imagenet-1k", "cifar10", "zoolake", "lar"])
    parser.add_argument("--results_dir", type=str, default="visualization/benchmark_vm_results")
    parser.add_argument("--speed_dir", type=str, default="visualization/speed_vm_results")
    parser.add_argument("--output_dir", type=str, default="visualization/sdr_results")
    return parser.parse_args()


def main():
    args = parse_args()
    compute_sdr_for_configuration(args.model, args.prune_method, args.drop_num, args.datasets, args.results_dir,
                                  args.speed_dir, args.output_dir)


if __name__ == "__main__":
    main()
