from __future__ import annotations

import os
import tempfile
from pathlib import Path

import torch


def load_detection_model(weights: Path, device: torch.device) -> torch.nn.Module:
    # 延迟导入，避免查看 CLI 帮助时触发 Ultralytics 的用户配置初始化。
    os.environ.setdefault("YOLO_CONFIG_DIR", str(Path(tempfile.gettempdir()) / "yolo_benchmark_ultralytics"))
    from ultralytics import YOLO

    wrapper = YOLO(str(weights))
    if wrapper.task != "detect":
        raise ValueError(f"仅支持目标检测模型，当前任务为：{wrapper.task}")
    model = wrapper.model.to(device).eval().float()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model


def unwrap_prediction(output: object) -> torch.Tensor:
    if isinstance(output, torch.Tensor):
        return output
    if isinstance(output, (tuple, list)) and output:
        return unwrap_prediction(output[0])
    raise TypeError(f"无法识别模型输出类型：{type(output)!r}")


class ExportWrapper(torch.nn.Module):
    def __init__(self, model: torch.nn.Module) -> None:
        super().__init__()
        self.model = model

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return unwrap_prediction(self.model(images))
