"""
Bridge connecting the Streamlit dashboard to the compiled native Parallel binary.
"""

import os
import subprocess
import json
from typing import Dict, Any, Optional

CLI_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "build", "bin", "parallel.exe"))

def get_system_info() -> Dict[str, Any]:
    if not os.path.exists(CLI_PATH):
        return {
            "application": "Parallel",
            "version": "1.0.0",
            "cpu_model": "Unknown CPU",
            "gpu_model": "N/A",
            "cuda_driver_version": "N/A",
            "cuda_available": False,
            "cuda_compiled": False,
            "binary_exists": False
        }
    
    try:
        proc = subprocess.run([CLI_PATH, "info", "--json"], capture_output=True, text=True, check=True)
        data = json.loads(proc.stdout)
        data["binary_exists"] = True
        return data
    except Exception as e:
        return {
            "application": "Parallel",
            "version": "1.0.0",
            "cpu_model": "Error reading CPU",
            "gpu_model": "N/A",
            "cuda_driver_version": "N/A",
            "cuda_available": False,
            "cuda_compiled": False,
            "binary_exists": True,
            "error": str(e)
        }

def process_image(
    input_path: str,
    output_path: str,
    operation: str,
    backend: str = "cpu",
    kernel_size: int = 3,
    sigma: float = 1.0
) -> Dict[str, Any]:
    if not os.path.exists(CLI_PATH):
        raise FileNotFoundError(f"Parallel binary not found at {CLI_PATH}. Run 'cmake --build build' first.")

    cmd = [
        CLI_PATH, "process",
        "--input", input_path,
        "--output", output_path,
        "--operation", operation,
        "--backend", backend,
        "--kernel-size", str(kernel_size),
        "--sigma", str(sigma),
        "--json"
    ]

    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(f"Processing failed (code {proc.returncode}):\n{proc.stderr}\n{proc.stdout}")

    try:
        return json.loads(proc.stdout)
    except json.JSONDecodeError:
        return {
            "success": True,
            "raw_output": proc.stdout
        }

def run_benchmark(
    input_path: Optional[str],
    operation: str = "all",
    backend: str = "cpu",
    iterations: int = 20,
    warmup: int = 3,
    output_csv: str = "results/benchmark.csv"
) -> bool:
    if not os.path.exists(CLI_PATH):
        raise FileNotFoundError(f"Parallel binary not found at {CLI_PATH}")

    cmd = [
        CLI_PATH, "benchmark",
        "--operation", operation,
        "--backend", backend,
        "--iterations", str(iterations),
        "--warmup", str(warmup),
        "--output", output_csv
    ]
    if input_path and os.path.exists(input_path):
        cmd.extend(["--input", input_path])

    proc = subprocess.run(cmd, capture_output=True, text=True)
    return proc.returncode == 0
