from __future__ import annotations

import importlib.metadata
import os
import platform
from typing import Any

import torch


class GpuMemoryMonitor:
    """用 NVML 采样当前进程显存；峰值是每批结束后的观测峰值。"""

    def __init__(self, device_id: int) -> None:
        try:
            import pynvml

            pynvml.nvmlInit()
            self.nvml = pynvml
            self.handle = pynvml.nvmlDeviceGetHandleByIndex(device_id)
            self.pid = os.getpid()
            self.peak_mib = 0.0
        except Exception as error:
            raise RuntimeError(f"NVML 初始化失败，无法按要求记录 GPU 显存：{error}") from error

    def current_mib(self) -> float:
        try:
            processes = list(self.nvml.nvmlDeviceGetComputeRunningProcesses(self.handle))
            try:
                processes += list(self.nvml.nvmlDeviceGetGraphicsRunningProcesses(self.handle))
            except self.nvml.NVMLError:
                pass
            used = max((int(item.usedGpuMemory) for item in processes if item.pid == self.pid), default=0)
        except Exception as error:
            raise RuntimeError(f"NVML 无法读取当前进程显存：{error}") from error
        value = used / (1024**2)
        self.peak_mib = max(self.peak_mib, value)
        return value

    def close(self) -> None:
        self.nvml.nvmlShutdown()


def collect_environment(device_id: int) -> dict[str, Any]:
    packages = [
        "torch",
        "torchvision",
        "ultralytics",
        "onnx",
        "onnxruntime-gpu",
        "tensorrt",
        "numpy",
        "opencv-python",
        "pycocotools",
    ]
    versions: dict[str, str | None] = {}
    for package in packages:
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = None
    gpu: dict[str, Any] = {"cuda_available": torch.cuda.is_available()}
    if torch.cuda.is_available() and device_id < torch.cuda.device_count():
        properties = torch.cuda.get_device_properties(device_id)
        gpu.update(
            {
                "device_id": device_id,
                "name": properties.name,
                "total_memory_mib": round(properties.total_memory / (1024**2), 2),
                "compute_capability": f"{properties.major}.{properties.minor}",
                "torch_cuda": torch.version.cuda,
                "cudnn": torch.backends.cudnn.version(),
            }
        )
        try:
            import pynvml

            pynvml.nvmlInit()
            gpu["driver"] = pynvml.nvmlSystemGetDriverVersion()
            pynvml.nvmlShutdown()
        except Exception as error:
            gpu["nvml_error"] = str(error)
    return {
        "platform": platform.platform(),
        "python": platform.python_version(),
        "processor": platform.processor(),
        "packages": versions,
        "gpu": gpu,
    }
