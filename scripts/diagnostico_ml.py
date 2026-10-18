#!/usr/bin/env mvtec-cnn
"""Diagnóstico de recursos para proyectos de ML.

Revisa sistema, memoria, disco, GPU y librerías, mide la velocidad de una
multiplicación de matrices y guarda todo en un JSON.
Uso: python diagnostico_ml.py
"""

import json
import os
import platform
import shutil
import statistics
import subprocess
import sys
import time
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

import numpy as np
import psutil
import torch

PACKAGES = [
    "torch",
    "torchvision",
    "numpy",
    "opencv-python",
    "scikit-learn",
]


def gb(n_bytes):
    return round(n_bytes / 2**30, 2)


def run(cmd):
    try:
        p = subprocess.run(cmd, capture_output=True, text=True, timeout=8)
        return p.stdout.strip() if p.returncode == 0 else None
    except (OSError, subprocess.TimeoutExpired):
        return None


def installed_version(pkg):
    """Versión instalada del paquete, o None si no está instalado."""
    try:
        return version(pkg)
    except PackageNotFoundError:
        return None


def time_matmul(a, b, sync=lambda: None, repeats=5):
    """Mediana en segundos de a @ b. La primera ejecución es de calentamiento."""
    a @ b
    sync()
    times = []
    for _ in range(repeats):
        start = time.perf_counter()
        a @ b
        sync()  # en GPU hay que esperar a que la operación termine
        times.append(time.perf_counter() - start)
    return round(statistics.median(times), 4)


def main():
    vm = psutil.virtual_memory()
    disk = shutil.disk_usage(Path.cwd())
    cuda = torch.cuda.is_available()
    mps = torch.backends.mps.is_available()

    result = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "system": {
            "os": platform.platform(),
            "architecture": platform.machine(),
            "python": sys.version.split()[0],
            "cpu": platform.processor() or "unknown",
            "logical_cores": os.cpu_count(),
            "physical_cores": psutil.cpu_count(logical=False),
        },
        "memory": {"total_gb": gb(vm.total), "available_gb": gb(vm.available)},
        "storage": {"total_gb": gb(disk.total), "free_gb": gb(disk.free)},
        "accelerators": {"pytorch_cuda_available": cuda, "pytorch_mps_available": mps},
        "packages": {pkg: installed_version(pkg) for pkg in PACKAGES},
        "benchmarks": {},
    }

    # --- GPU ---
    acc = result["accelerators"]

    if cuda:
        acc["cuda_devices"] = []
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            acc["cuda_devices"].append(
                {
                    "name": props.name,
                    "vram_gb": gb(props.total_memory),
                    # Cada multiprocesador (SM) agrupa 64–128 núcleos CUDA según la generación
                    "multiprocessors": props.multi_processor_count,
                    "compute_capability": f"{props.major}.{props.minor}",
                }
            )

    if platform.system() == "Darwin":
        acc["apple_chip_info"] = run(["sysctl", "-n", "machdep.cpu.brand_string"])
        # Núcleos de CPU de rendimiento (P) y de eficiencia (E)
        result["system"]["performance_cores"] = run(
            ["sysctl", "-n", "hw.perflevel0.physicalcpu"]
        )
        result["system"]["efficiency_cores"] = run(
            ["sysctl", "-n", "hw.perflevel1.physicalcpu"]
        )
        # Núcleos de la GPU integrada (Apple Silicon)
        gpu_info = run(["system_profiler", "SPDisplaysDataType"]) or ""
        for line in gpu_info.splitlines():
            if "Total Number of Cores" in line:
                acc["apple_gpu_cores"] = int(line.split(":")[1])

    nvidia = run(
        ["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"]
    )

    if nvidia:
        acc["nvidia_smi"] = nvidia.splitlines()

    # --- Benchmark: multiplicar dos matrices 1024×1024 ---
    size = 1024
    bench = result["benchmarks"]

    rng = np.random.default_rng(42)
    a = rng.random((size, size), dtype=np.float32)
    b = rng.random((size, size), dtype=np.float32)
    bench["numpy_matmul_cpu"] = {
        "matrix_size": size,
        "repeats": 5,
        "median_seconds": time_matmul(a, b),
    }

    syncs = {
        "cpu": lambda: None,
        "cuda": lambda: torch.cuda.synchronize(),
        "mps": lambda: torch.mps.synchronize(),
    }
    devices = ["cpu"] + (["cuda"] if cuda else []) + (["mps"] if mps else [])
    torch.manual_seed(42)
    for device in devices:
        try:
            a = torch.randn(size, size, device=device)
            b = torch.randn(size, size, device=device)
            bench[f"torch_matmul_{device}"] = {
                "matrix_size": size,
                "repeats": 5,
                "median_seconds": time_matmul(a, b, syncs[device]),
            }
        except RuntimeError as exc:
            bench[f"torch_{device}_error"] = str(exc)

    # --- Guardar y mostrar ---
    output = Path("results/system_diagnostics/diagnostico_sistema.json")
    output.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(result, indent=2, ensure_ascii=False)
    output.write_text(text, encoding="utf-8")
    print(f"\nDiagnóstico guardado en: {output.resolve()}")


if __name__ == "__main__":
    main()
