from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
import re
import subprocess
import sys


@dataclass
class ProbeResult:
    timestamp: str
    cpu_model: str
    cpu_cores: int
    cpu_threads: int
    cpu_max_mhz: float
    mem_total_gb: float
    mem_available_gb: float
    gpu_name: str
    gpu_mem_total_gb: float
    gpu_mem_free_gb: float
    torch_version: str
    torch_cuda_available: bool
    torch_cuda_devices: int


def _run(cmd: list[str]) -> str:
    try:
        out = subprocess.check_output(cmd, stderr=subprocess.STDOUT, text=True)
    except Exception:
        return ""
    return out.strip()


def _parse_lscpu(text: str) -> tuple[str, int, int, float]:
    model = "unknown"
    cores = 0
    threads = 0
    max_mhz = 0.0
    for line in text.splitlines():
        if "Model name:" in line:
            model = line.split(":", 1)[1].strip()
        elif re.match(r"^Core\(s\) per socket:\s+\d+", line):
            cores = int(line.split(":", 1)[1].strip())
        elif re.match(r"^CPU\(s\):\s+\d+", line):
            threads = int(line.split(":", 1)[1].strip())
        elif "CPU max MHz:" in line:
            try:
                max_mhz = float(line.split(":", 1)[1].strip())
            except ValueError:
                max_mhz = 0.0
    return model, cores, threads, max_mhz


def _parse_meminfo(text: str) -> tuple[float, float]:
    total = 0.0
    available = 0.0
    for line in text.splitlines():
        if line.startswith("MemTotal:"):
            total = float(line.split()[1]) / 1024 / 1024
        elif line.startswith("MemAvailable:"):
            available = float(line.split()[1]) / 1024 / 1024
    return total, available


def _parse_nvidia(text: str) -> tuple[str, float, float]:
    if not text:
        return "N/A", 0.0, 0.0
    line = text.splitlines()[0]
    parts = [x.strip() for x in line.split(",")]
    if len(parts) < 5:
        return "N/A", 0.0, 0.0

    name = parts[1]
    mem_total = float(parts[3].split()[0]) / 1024
    mem_free = float(parts[4].split()[0]) / 1024
    return name, mem_total, mem_free


def _torch_info() -> tuple[str, bool, int]:
    try:
        import torch

        return torch.__version__, torch.cuda.is_available(), torch.cuda.device_count()
    except Exception:
        return "not-installed", False, 0


def probe() -> ProbeResult:
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    lscpu = _run(["lscpu"])
    meminfo = Path("/proc/meminfo").read_text(encoding="utf-8")
    nvidia = _run(
        [
            "nvidia-smi",
            "--query-gpu=index,name,driver_version,memory.total,memory.free",
            "--format=csv,noheader",
        ]
    )

    cpu_model, cpu_cores, cpu_threads, cpu_max_mhz = _parse_lscpu(lscpu)
    mem_total_gb, mem_available_gb = _parse_meminfo(meminfo)
    gpu_name, gpu_mem_total_gb, gpu_mem_free_gb = _parse_nvidia(nvidia)
    torch_version, torch_cuda_available, torch_cuda_devices = _torch_info()

    return ProbeResult(
        timestamp=timestamp,
        cpu_model=cpu_model,
        cpu_cores=cpu_cores,
        cpu_threads=cpu_threads,
        cpu_max_mhz=cpu_max_mhz,
        mem_total_gb=mem_total_gb,
        mem_available_gb=mem_available_gb,
        gpu_name=gpu_name,
        gpu_mem_total_gb=gpu_mem_total_gb,
        gpu_mem_free_gb=gpu_mem_free_gb,
        torch_version=torch_version,
        torch_cuda_available=torch_cuda_available,
        torch_cuda_devices=torch_cuda_devices,
    )


def suggest_settings(result: ProbeResult) -> dict:
    envs = min(max(result.cpu_threads // 2, 16), 24)
    threads = min(max(result.cpu_threads - 8, 8), result.cpu_threads)
    batch_size = 1024 if result.gpu_mem_total_gb >= 20 else 512

    return {
        "runtime": {
            "detected_at": result.timestamp,
            "cpu_model": result.cpu_model,
            "gpu_name": result.gpu_name,
            "memory_total_gb": round(result.mem_total_gb, 2),
            "memory_available_gb": round(result.mem_available_gb, 2),
            "env_concurrency": envs,
            "rust_num_threads": threads,
            "mcts_workers": min(20, threads),
            "train_batch_size": batch_size,
            "replay_capacity": 1_000_000,
            "ui_fps": 20,
            "amp": True,
            "torch_compile": True,
        }
    }


def write_outputs(result: ProbeResult, settings: dict, root: Path) -> None:
    profile_path = root / "docs/system_profile/2026-02-09-hardware-baseline.md"
    config_path = root / "configs/runtime/hardware.auto.toml"

    profile_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.parent.mkdir(parents=True, exist_ok=True)

    profile = f"""# Hardware Baseline (2026-02-09)

- Probe Time: {result.timestamp}
- CPU: {result.cpu_model}
- CPU Cores/Threads: {result.cpu_cores}/{result.cpu_threads}
- CPU Max MHz: {result.cpu_max_mhz:.1f}
- RAM Total/Available: {result.mem_total_gb:.2f}GB / {result.mem_available_gb:.2f}GB
- GPU: {result.gpu_name}
- GPU VRAM Total/Free: {result.gpu_mem_total_gb:.2f}GB / {result.gpu_mem_free_gb:.2f}GB
- PyTorch: {result.torch_version}
- CUDA Available: {result.torch_cuda_available} ({result.torch_cuda_devices} device)

## Recommended Runtime Cap

- env_concurrency: {settings['runtime']['env_concurrency']}
- rust_num_threads: {settings['runtime']['rust_num_threads']}
- mcts_workers: {settings['runtime']['mcts_workers']}
- train_batch_size: {settings['runtime']['train_batch_size']}
- replay_capacity: {settings['runtime']['replay_capacity']}
- ui_fps: {settings['runtime']['ui_fps']}
- amp: {settings['runtime']['amp']}
- torch_compile: {settings['runtime']['torch_compile']}
"""
    profile_path.write_text(profile, encoding="utf-8")

    toml_text = f"""[runtime]
detected_at = \"{settings['runtime']['detected_at']}\"
cpu_model = \"{settings['runtime']['cpu_model']}\"
gpu_name = \"{settings['runtime']['gpu_name']}\"
memory_total_gb = {settings['runtime']['memory_total_gb']}
memory_available_gb = {settings['runtime']['memory_available_gb']}
env_concurrency = {settings['runtime']['env_concurrency']}
rust_num_threads = {settings['runtime']['rust_num_threads']}
mcts_workers = {settings['runtime']['mcts_workers']}
train_batch_size = {settings['runtime']['train_batch_size']}
replay_capacity = {settings['runtime']['replay_capacity']}
ui_fps = {settings['runtime']['ui_fps']}
amp = {str(settings['runtime']['amp']).lower()}
torch_compile = {str(settings['runtime']['torch_compile']).lower()}
"""
    config_path.write_text(toml_text, encoding="utf-8")


def main() -> int:
    root = Path(__file__).resolve().parents[2]
    result = probe()
    settings = suggest_settings(result)
    write_outputs(result, settings, root)

    print("[probe] hardware profile written")
    print(f"[probe] cpu={result.cpu_model} threads={result.cpu_threads}")
    print(f"[probe] gpu={result.gpu_name} vram={result.gpu_mem_total_gb:.2f}GB")
    return 0


if __name__ == "__main__":
    sys.exit(main())
