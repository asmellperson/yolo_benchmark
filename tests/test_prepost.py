import numpy as np
import pytest
import torch

from yolo_benchmark.data import ImageMeta, letterbox
from yolo_benchmark.postprocess import normalize_prediction_shape, postprocess


def test_letterbox_shape_and_ratio() -> None:
    image = np.zeros((100, 200, 3), dtype=np.uint8)
    output, ratio, pad = letterbox(image, (640, 640))
    assert output.shape == (640, 640, 3)
    assert ratio == 3.2
    assert pad == (0.0, 160.0)


def test_prediction_shape_transpose() -> None:
    prediction = torch.zeros((1, 84, 8400))
    assert normalize_prediction_shape(prediction).shape == (1, 8400, 84)


def test_postprocess_scales_boxes_to_original_image() -> None:
    # 一个类别、一个候选框：[中心 x, 中心 y, 宽, 高, 分类分数]
    prediction = torch.tensor([[[320.0], [320.0], [320.0], [160.0], [0.9]]])
    meta = ImageMeta(image_id=1, original_shape=(100, 200), ratio=3.2, pad=(0.0, 160.0))
    result = postprocess(prediction, [meta], confidence=0.1, iou=0.7, max_detections=10)[0]
    expected = torch.tensor([50.0, 25.0, 150.0, 75.0])
    assert torch.allclose(result[0, :4], expected)
    assert result[0, 4].item() == pytest.approx(0.9)
