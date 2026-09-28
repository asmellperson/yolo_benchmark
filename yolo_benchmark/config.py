from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class ModelConfig:
    weights: Path
    onnx: Path
    engine: Path


@dataclass(frozen=True)
class InputConfig:
    size: tuple[int, int]
    batch_size: int


@dataclass(frozen=True)
class DatasetConfig:
    images: Path
    annotations: Path
    limit: int | None


@dataclass(frozen=True)
class PostprocessConfig:
    confidence: float
    iou: float
    max_detections: int


@dataclass(frozen=True)
class BenchmarkConfig:
    warmup_iterations: int
    rounds: int
    iterations_per_round: int
    seed: int


@dataclass(frozen=True)
class RuntimeConfig:
    device_id: int
    trt_workspace_gib: int
    output_dir: Path


@dataclass(frozen=True)
class Config:
    root: Path
    model: ModelConfig
    input: InputConfig
    dataset: DatasetConfig
    postprocess: PostprocessConfig
    benchmark: BenchmarkConfig
    runtime: RuntimeConfig

    def as_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data.pop("root")

        def convert(value: Any) -> Any:
            if isinstance(value, Path):
                return str(value)
            if isinstance(value, dict):
                return {k: convert(v) for k, v in value.items()}
            if isinstance(value, (tuple, list)):
                return [convert(v) for v in value]
            return value

        return convert(data)


def _path(root: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else root / path


def load_config(path: str | Path) -> Config:
    config_path = Path(path).resolve()
    with config_path.open("r", encoding="utf-8") as file:
        raw = yaml.safe_load(file)
    root = config_path.parent.parent

    size = tuple(int(v) for v in raw["input"]["size"])
    if len(size) != 2 or min(size) <= 0:
        raise ValueError("input.size 必须是两个正整数：[高, 宽]")
    batch_size = int(raw["input"]["batch_size"])
    if batch_size <= 0:
        raise ValueError("input.batch_size 必须大于 0")

    return Config(
        root=root,
        model=ModelConfig(
            weights=_path(root, raw["model"]["weights"]),
            onnx=_path(root, raw["model"]["onnx"]),
            engine=_path(root, raw["model"]["engine"]),
        ),
        input=InputConfig(size=size, batch_size=batch_size),
        dataset=DatasetConfig(
            images=_path(root, raw["dataset"]["images"]),
            annotations=_path(root, raw["dataset"]["annotations"]),
            limit=raw["dataset"].get("limit"),
        ),
        postprocess=PostprocessConfig(**raw["postprocess"]),
        benchmark=BenchmarkConfig(**raw["benchmark"]),
        runtime=RuntimeConfig(
            device_id=int(raw["runtime"]["device_id"]),
            trt_workspace_gib=int(raw["runtime"]["trt_workspace_gib"]),
            output_dir=_path(root, raw["runtime"]["output_dir"]),
        ),
    )
