"""
Automated Python test suite for Parallel benchmarking, parsing, and CLI verification.
"""

import os
import subprocess
import json
import pytest
import pandas as pd
from scripts.plot_results import plot_benchmarks
from scripts.generate_report import generate_report

CLI_EXE = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "build", "bin", "parallel.exe"))

def test_cli_executable_exists():
    assert os.path.exists(CLI_EXE), f"Parallel binary not found at {CLI_EXE}"

def test_cli_info_json():
    result = subprocess.run([CLI_EXE, "info", "--json"], capture_output=True, text=True, check=True)
    data = json.loads(result.stdout)
    assert data["application"] == "Parallel"
    assert "cpu_model" in data
    assert "version" in data
    assert isinstance(data["cuda_available"], bool)

def test_cli_process_synthetic_json():
    result = subprocess.run([
        CLI_EXE, "process",
        "--synthetic", "256x256",
        "--op", "grayscale",
        "-b", "cpu",
        "--json"
    ], capture_output=True, text=True, check=True)
    
    data = json.loads(result.stdout)
    assert data["success"] is True
    assert data["operation"] == "grayscale"
    assert data["backend"] == "cpu"
    assert data["width"] == 256
    assert data["height"] == 256
    assert data["execution_time_ms"] > 0.0

def test_benchmark_csv_schema(tmp_path):
    csv_file = tmp_path / "test_benchmark.csv"
    result = subprocess.run([
        CLI_EXE, "benchmark",
        "--synthetic", "128x128",
        "--op", "grayscale",
        "-b", "cpu",
        "-n", "5",
        "-w", "1",
        "-o", str(csv_file)
    ], capture_output=True, text=True, check=True)
    
    assert os.path.exists(csv_file)
    df = pd.read_csv(csv_file)
    assert not df.empty
    expected_cols = [
        "timestamp", "operation", "backend", "width", "height", "channels",
        "pixel_count", "iterations", "cpu_median_ms", "cpu_min_ms", "cpu_max_ms"
    ]
    for col in expected_cols:
        assert col in df.columns
    assert df["width"].iloc[0] == 128
    assert df["height"].iloc[0] == 128
    assert df["iterations"].iloc[0] == 5

def test_plot_generation(tmp_path):
    csv_file = tmp_path / "data.csv"
    out_dir = tmp_path / "plots"
    # Create valid synthetic benchmark CSV
    df = pd.DataFrame([{
        "timestamp": "2026-10-22T10:00:00",
        "operation": "blur",
        "backend": "cpu",
        "width": 512,
        "height": 512,
        "channels": 3,
        "pixel_count": 262144,
        "iterations": 10,
        "cpu_median_ms": 12.5,
        "cpu_min_ms": 11.8,
        "cpu_max_ms": 14.1,
        "gpu_kernel_median_ms": 0.0,
        "gpu_kernel_min_ms": 0.0,
        "gpu_kernel_max_ms": 0.0,
        "gpu_h2d_ms": 0.0,
        "gpu_d2h_ms": 0.0,
        "gpu_pipeline_median_ms": 0.0,
        "speedup_kernel": 0.0,
        "speedup_pipeline": 0.0,
        "cpu_model": "Test CPU",
        "gpu_model": "N/A",
        "cuda_version": "N/A"
    }])
    df.to_csv(csv_file, index=False)
    success = plot_benchmarks(str(csv_file), str(out_dir))
    assert success is True
    assert os.path.exists(out_dir / "benchmark_by_operation.png")

def test_empty_csv_handling(tmp_path):
    empty_csv = tmp_path / "empty.csv"
    empty_csv.write_text("")
    assert plot_benchmarks(str(empty_csv), str(tmp_path)) is False
    assert generate_report(str(empty_csv), str(tmp_path / "rep.md")) is False

def test_missing_csv_handling():
    assert plot_benchmarks("non_existent.csv") is False
    assert generate_report("non_existent.csv") is False
