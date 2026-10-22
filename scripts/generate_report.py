"""
Parallel Automated Benchmark Report Generator
Compiles measured benchmark CSV data into a clean, markdown technical report.
"""

import os
import argparse
import pandas as pd
from datetime import datetime

def generate_report(csv_path="results/benchmark.csv", output_path="results/benchmark_report.md"):
    if not os.path.exists(csv_path):
        print(f"[Error] Benchmark file not found: {csv_path}")
        return False

    try:
        df = pd.read_csv(csv_path)
    except Exception as e:
        print(f"[Error] Could not read CSV: {e}")
        return False

    if df.empty:
        print("[Warning] Benchmark CSV is empty.")
        return False

    cpu_model = df['cpu_model'].iloc[0] if 'cpu_model' in df.columns else 'Unknown'
    gpu_model = df['gpu_model'].iloc[0] if 'gpu_model' in df.columns else 'Unknown'
    cuda_ver = df['cuda_version'].iloc[0] if 'cuda_version' in df.columns else 'N/A'
    has_gpu = (df['gpu_kernel_median_ms'] > 0).any() if 'gpu_kernel_median_ms' in df.columns else False

    report = []
    report.append("# Parallel: Performance Benchmark Report\n")
    report.append(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}  \n")
    report.append(f"**CPU Architecture:** `{cpu_model}`  \n")
    report.append(f"**GPU Architecture:** `{gpu_model}`  \n")
    report.append(f"**CUDA Runtime/Driver:** `{cuda_ver}`  \n")
    report.append(f"**Active Mode:** `{'CUDA Accelerated' if has_gpu else 'CPU Reference Mode'}`  \n\n")

    report.append("## 1. Summary of Measured Operations\n")
    report.append("| Operation | Resolution | Pixels | Repetitions | CPU Median (ms) | CPU Min (ms) | CPU Max (ms) |")
    if has_gpu:
        report[-1] += " GPU Kernel (ms) | GPU Pipeline (ms) | Speedup (Pipeline) |"
    report.append("| :--- | :--- | :--- | :--- | :--- | :--- | :--- |")
    if has_gpu:
        report[-1] += " :--- | :--- | :--- |"

    for _, row in df.iterrows():
        op = str(row['operation']).capitalize()
        res = f"{row['width']}x{row['height']}"
        pixels = f"{row['pixel_count']:,}"
        iters = row['iterations']
        cpu_med = f"{row['cpu_median_ms']:.3f}"
        cpu_min = f"{row['cpu_min_ms']:.3f}"
        cpu_max = f"{row['cpu_max_ms']:.3f}"

        line = f"| {op} | {res} | {pixels} | {iters} | {cpu_med} | {cpu_min} | {cpu_max} |"
        if has_gpu:
            k_med = f"{row['gpu_kernel_median_ms']:.3f}"
            p_med = f"{row['gpu_pipeline_median_ms']:.3f}"
            spd = f"{row['speedup_pipeline']:.2f}x"
            line += f" {k_med} | {p_med} | {spd} |"
        report.append(line)

    report.append("\n## 2. Measurement Methodology\n")
    report.append("- **CPU Monotonic Clock:** High-resolution steady clock (`std::chrono::high_resolution_clock`).")
    report.append("- **Warmup Phase:** Unrecorded iterations executed prior to measurement to prime caches and instruction pipelines.")
    report.append("- **Statistical Metric:** Median execution time across configured repetitions to minimize transient OS scheduling noise.")
    report.append("- **GPU Metrics (when active):** Explicitly isolates device global memory transfers (H2D, D2H) from pure compute kernel execution using CUDA hardware events.")

    report_text = "\n".join(report) + "\n"
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(report_text)

    print(f"Report generated: {output_path}")
    return True

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate benchmark report")
    parser.add_argument("--csv", default="results/benchmark.csv", help="Path to benchmark CSV")
    parser.add_argument("--output", default="results/benchmark_report.md", help="Output markdown path")
    args = parser.parse_args()

    generate_report(args.csv, args.output)
