from __future__ import annotations

import gc
import json
from pathlib import Path
from typing import Any

import torch
from tqdm import tqdm

from .backends import create_backend
from .config import Config
from .data import iter_padded_batches, load_coco_samples, preprocess_batch, read_image
from .postprocess import postprocess


COCO80_TO_91 = [
    1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 14, 15, 16, 17, 18, 19, 20, 21,
    22, 23, 24, 25, 27, 28, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44,
    46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65,
    67, 70, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 84, 85, 86, 87, 88, 89, 90,
]


def evaluate_coco(config: Config, backend_names: list[str], output_dir: Path) -> dict[str, Any]:
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval

    samples = load_coco_samples(config.dataset.images, config.dataset.annotations, config.dataset.limit)
    ground_truth = COCO(str(config.dataset.annotations))
    selected_ids = [sample.image_id for sample in samples]
    output_dir.mkdir(parents=True, exist_ok=True)
    all_metrics: dict[str, Any] = {}

    for backend_key in backend_names:
        backend = create_backend(backend_key, config)
        predictions: list[dict[str, Any]] = []
        for batch, valid_count in tqdm(
            iter_padded_batches(samples, config.input.batch_size),
            total=(len(samples) + config.input.batch_size - 1) // config.input.batch_size,
            desc=f"COCO {backend.name}",
        ):
            images = [read_image(sample) for sample in batch]
            cpu_tensor, metas = preprocess_batch(batch, images, config.input.size)
            gpu_tensor = cpu_tensor.to(backend.device, dtype=backend.input_dtype).contiguous()
            raw = backend.forward(gpu_tensor)
            detections = postprocess(
                raw,
                metas,
                config.postprocess.confidence,
                config.postprocess.iou,
                config.postprocess.max_detections,
            )
            for sample, result in zip(batch[:valid_count], detections[:valid_count], strict=True):
                for x1, y1, x2, y2, score, label in result.cpu().tolist():
                    class_index = int(label)
                    predictions.append(
                        {
                            "image_id": sample.image_id,
                            "category_id": COCO80_TO_91[class_index],
                            "bbox": [x1, y1, x2 - x1, y2 - y1],
                            "score": score,
                        }
                    )
        backend.synchronize()
        prediction_path = output_dir / f"coco_predictions_{backend.name}.json"
        prediction_path.write_text(json.dumps(predictions), encoding="utf-8")
        if not predictions:
            raise RuntimeError(f"{backend.name} 在 COCO 数据上没有产生任何预测")

        detected = ground_truth.loadRes(str(prediction_path))
        evaluator = COCOeval(ground_truth, detected, "bbox")
        evaluator.params.imgIds = selected_ids
        evaluator.evaluate()
        evaluator.accumulate()
        evaluator.summarize()
        values = evaluator.stats.tolist()
        all_metrics[backend.name] = {
            "AP_50_95": values[0],
            "AP_50": values[1],
            "AP_75": values[2],
            "AP_small": values[3],
            "AP_medium": values[4],
            "AP_large": values[5],
            "AR_1": values[6],
            "AR_10": values[7],
            "AR_100": values[8],
            "evaluated_images": len(selected_ids),
        }
        del backend
        gc.collect()
        torch.cuda.empty_cache()

    (output_dir / "coco_metrics.json").write_text(
        json.dumps(all_metrics, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return all_metrics

