from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Sequence

import cv2
import numpy as np
import torch


@dataclass(frozen=True)
class Sample:
    image_id: int
    path: Path


@dataclass(frozen=True)
class ImageMeta:
    image_id: int
    original_shape: tuple[int, int]
    ratio: float
    pad: tuple[float, float]


def load_coco_samples(images_dir: Path, annotations: Path, limit: int | None = None) -> list[Sample]:
    if not images_dir.is_dir():
        raise FileNotFoundError(f"COCO 图片目录不存在：{images_dir}")
    if not annotations.is_file():
        raise FileNotFoundError(f"COCO 标注文件不存在：{annotations}")
    with annotations.open("r", encoding="utf-8") as file:
        records = json.load(file)["images"]
    samples = [Sample(int(item["id"]), images_dir / item["file_name"]) for item in records]
    samples.sort(key=lambda item: item.image_id)
    if limit is not None:
        samples = samples[: int(limit)]
    missing = [str(item.path) for item in samples if not item.path.is_file()]
    if missing:
        raise FileNotFoundError(f"有 {len(missing)} 张图片缺失，首个文件：{missing[0]}")
    return samples


def read_image(sample: Sample) -> np.ndarray:
    image = cv2.imread(str(sample.path), cv2.IMREAD_COLOR)
    if image is None:
        raise RuntimeError(f"图片读取失败：{sample.path}")
    return image


def letterbox(image: np.ndarray, size: tuple[int, int]) -> tuple[np.ndarray, float, tuple[float, float]]:
    """按 YOLO 的居中 letterbox 规则缩放，返回 BGR 图、缩放比例和半边填充。"""
    target_h, target_w = size
    height, width = image.shape[:2]
    ratio = min(target_h / height, target_w / width)
    resized_w, resized_h = round(width * ratio), round(height * ratio)
    if (resized_w, resized_h) != (width, height):
        image = cv2.resize(image, (resized_w, resized_h), interpolation=cv2.INTER_LINEAR)

    pad_w = (target_w - resized_w) / 2
    pad_h = (target_h - resized_h) / 2
    left, right = round(pad_w - 0.1), round(pad_w + 0.1)
    top, bottom = round(pad_h - 0.1), round(pad_h + 0.1)
    image = cv2.copyMakeBorder(image, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(114, 114, 114))
    return image, ratio, (pad_w, pad_h)


def preprocess_batch(
    samples: Sequence[Sample], images: Sequence[np.ndarray], size: tuple[int, int]
) -> tuple[torch.Tensor, list[ImageMeta]]:
    arrays: list[np.ndarray] = []
    metas: list[ImageMeta] = []
    for sample, image in zip(samples, images, strict=True):
        padded, ratio, pad = letterbox(image, size)
        rgb = cv2.cvtColor(padded, cv2.COLOR_BGR2RGB)
        arrays.append(np.ascontiguousarray(rgb.transpose(2, 0, 1), dtype=np.float32) / 255.0)
        metas.append(ImageMeta(sample.image_id, image.shape[:2], ratio, pad))
    return torch.from_numpy(np.stack(arrays)), metas


def iter_padded_batches(samples: Sequence[Sample], batch_size: int) -> Iterator[tuple[list[Sample], int]]:
    """最后一批复制末项补齐静态 batch，valid_count 用于丢弃补位输出。"""
    for start in range(0, len(samples), batch_size):
        batch = list(samples[start : start + batch_size])
        valid_count = len(batch)
        if valid_count < batch_size:
            batch.extend([batch[-1]] * (batch_size - valid_count))
        yield batch, valid_count


def cycle_batches(samples: Sequence[Sample], batch_size: int, iterations: int) -> Iterator[list[Sample]]:
    if not samples:
        raise ValueError("数据集为空")
    for index in range(iterations):
        start = (index * batch_size) % len(samples)
        yield [samples[(start + offset) % len(samples)] for offset in range(batch_size)]

