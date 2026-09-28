from __future__ import annotations

from collections.abc import Sequence

import torch
from torchvision.ops import batched_nms

from .data import ImageMeta


def _xywh_to_xyxy(boxes: torch.Tensor) -> torch.Tensor:
    output = torch.empty_like(boxes)
    half = boxes[:, 2:] / 2
    output[:, :2] = boxes[:, :2] - half
    output[:, 2:] = boxes[:, :2] + half
    return output


def normalize_prediction_shape(prediction: torch.Tensor) -> torch.Tensor:
    if prediction.ndim != 3:
        raise ValueError(f"期望三维 YOLO 输出，实际为 {tuple(prediction.shape)}")
    # YOLO11/YOLOv8 导出通常为 [B, 4+C, N]。
    if prediction.shape[2] < 5 <= prediction.shape[1] or prediction.shape[1] < prediction.shape[2]:
        prediction = prediction.transpose(1, 2)
    if prediction.shape[2] < 5:
        raise ValueError(f"YOLO 输出通道数异常：{tuple(prediction.shape)}")
    return prediction


def postprocess(
    prediction: torch.Tensor,
    metas: Sequence[ImageMeta],
    confidence: float,
    iou: float,
    max_detections: int,
) -> list[torch.Tensor]:
    """统一执行置信度过滤、按类别 NMS，并映射回原图坐标。"""
    # 后处理统一在 FP32 执行，避免 TensorRT 输出类型影响 NMS 数值口径。
    prediction = normalize_prediction_shape(prediction).float()
    results: list[torch.Tensor] = []
    for item, meta in zip(prediction, metas, strict=True):
        scores, labels = item[:, 4:].max(dim=1)
        keep = scores > confidence
        boxes = _xywh_to_xyxy(item[keep, :4])
        scores, labels = scores[keep], labels[keep]
        if boxes.numel():
            indices = batched_nms(boxes.float(), scores.float(), labels, iou)[:max_detections]
            boxes, scores, labels = boxes[indices], scores[indices], labels[indices]
            boxes[:, [0, 2]] -= meta.pad[0]
            boxes[:, [1, 3]] -= meta.pad[1]
            boxes /= meta.ratio
            height, width = meta.original_shape
            boxes[:, [0, 2]].clamp_(0, width)
            boxes[:, [1, 3]].clamp_(0, height)
            result = torch.cat((boxes, scores[:, None], labels[:, None].to(boxes.dtype)), dim=1)
        else:
            result = item.new_zeros((0, 6))
        results.append(result)
    return results
