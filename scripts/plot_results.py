"""
Parallel Benchmark Plotting & Visualization Utility
Generates publication-quality charts from benchmark CSV data.
"""

import os
import sys
import argparse
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

def plot_benchmarks(csv_path="results/benchmark.csv", output_dir="results"):
    if not os.path.exists(csv_path):
        print(f"[Error] Benchmark file not found: {csv_path}")
        return False

    try:
        df = pd.read_csv(csv_path)
    except Exception as e:
        print(f"[Error] Failed to read CSV: {e}")
        return False

    if df.empty:
        print("[Warning] Benchmark CSV is empty. No plots generated.")
        return False

    os.makedirs(output_dir, exist_ok=True)
    
    # Configure styling
    plt.style.use('seaborn-v0_8-darkgrid' if 'seaborn-v0_8-darkgrid' in plt.style.available else 'default')
    plt.rcParams['font.sans-serif'] = 'DejaVu Sans'
    plt.rcParams['axes.edgecolor'] = '#333333'
    plt.rcParams['axes.linewidth'] = 0.8

    has_gpu = (df['gpu_kernel_median_ms'] > 0).any()

    # 1. Plot Execution Time by Operation
    fig, ax = plt.subplots(figsize=(10, 6), dpi=300)
    ops = df['operation'].unique()
    x = range(len(ops))
    width = 0.35 if has_gpu else 0.5

    cpu_medians = [df[df['operation'] == op]['cpu_median_ms'].median() for op in ops]
    bars1 = ax.bar([i - (width/2 if has_gpu else 0) for i in x], cpu_medians, width, label='CPU (Median)', color='#00a8e8')

    if has_gpu:
        gpu_medians = [df[df['operation'] == op]['gpu_kernel_median_ms'].median() for op in ops]
        bars2 = ax.bar([i + width/2 for i in x], gpu_medians, width, label='GPU Kernel (Median)', color='#00c49f')

    ax.set_ylabel('Execution Time (ms)', fontsize=12, fontweight='bold')
    ax.set_title('Parallel Benchmark: Execution Time by Operation', fontsize=14, fontweight='bold', pad=15)
    ax.set_xticks(list(x))
    ax.set_xticklabels([op.capitalize() for op in ops], fontsize=11)
    ax.legend(frameon=True, facecolor='#ffffff', edgecolor='#cccccc')
    ax.grid(axis='y', linestyle='--', alpha=0.7)

    chart1_path = os.path.join(output_dir, "benchmark_by_operation.png")
    fig.tight_layout()
    fig.savefig(chart1_path)
    plt.close(fig)
    print(f"Generated chart: {chart1_path}")

    # 2. Plot Latency vs. Image Resolution / Pixel Count
    if len(df['pixel_count'].unique()) > 1:
        fig, ax = plt.subplots(figsize=(10, 6), dpi=300)
        for op in ops:
            sub = df[df['operation'] == op].sort_values(by='pixel_count')
            ax.plot(sub['pixel_count'] / 1e6, sub['cpu_median_ms'], marker='o', label=f"{op.capitalize()} (CPU)")
            if has_gpu:
                ax.plot(sub['pixel_count'] / 1e6, sub['gpu_pipeline_median_ms'], marker='s', linestyle='--', label=f"{op.capitalize()} (GPU Pipeline)")

        ax.set_xlabel('Resolution (Megapixels)', fontsize=12, fontweight='bold')
        ax.set_ylabel('Execution Time (ms)', fontsize=12, fontweight='bold')
        ax.set_title('Execution Time vs. Image Size', fontsize=14, fontweight='bold', pad=15)
        ax.legend(frameon=True)
        ax.grid(True, linestyle='--', alpha=0.7)

        chart2_path = os.path.join(output_dir, "benchmark_scaling.png")
        fig.tight_layout()
        fig.savefig(chart2_path)
        plt.close(fig)
        print(f"Generated chart: {chart2_path}")

    return True

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot Parallel benchmark results")
    parser.add_argument("--csv", default="results/benchmark.csv", help="Path to benchmark CSV")
    parser.add_argument("--output-dir", default="results", help="Directory to save generated charts")
    args = parser.parse_args()

    plot_benchmarks(args.csv, args.output_dir)
