from __future__ import annotations

import gc
import json
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as functional

from .backends import create_backend
from .config import Config
from .data import cycle_batches, load_coco_samples, preprocess_batch, read_image


def _cases(config: Config, random_cases: int, real_cases: int) -> list[tuple[str, torch.Tensor]]:
    generator = torch.Generator().manual_seed(config.benchmark.seed)
    shape = (config.input.batch_size, 3, *config.input.size)
    cases = [(f"random_{index}", torch.rand(shape, generator=generator)) for index in range(random_cases)]
    if real_cases and config.dataset.images.is_dir() and config.dataset.annotations.is_file():
        samples = load_coco_samples(config.dataset.images, config.dataset.annotations, config.dataset.limit)
        for index, batch in enumerate(cycle_batches(samples, config.input.batch_size, real_cases)):
            tensor, _ = preprocess_batch(batch, [read_image(item) for item in batch], config.input.size)
            cases.append((f"coco_{index}", tensor))
    return cases


def check_consistency(
    config: Config,
    target_names: list[str],
    output_dir: Path,
    random_cases: int = 2,
    real_cases: int = 2,
) -> dict[str, Any]:
    cases = _cases(config, random_cases, real_cases)
    reference = create_backend("torch", config)
    expected: dict[str, torch.Tensor] = {}
    for case_name, cpu_tensor in cases:
        tensor = cpu_tensor.to(reference.device, dtype=reference.input_dtype).contiguous()
        expected[case_name] = reference.forward(tensor).float().cpu()
    reference.synchronize()
    del reference
    gc.collect()
    torch.cuda.empty_cache()

    report: dict[str, Any] = {}
    for target_name in target_names:
        if target_name == "torch":
            continue
        target = create_backend(target_name, config)
        tolerance = {"onnx": (1e-4, 1e-4), "trt": (5e-2, 5e-2)}[target_name]
        case_reports = []
        for case_name, cpu_tensor in cases:
            tensor = cpu_tensor.to(target.device, dtype=target.input_dtype).contiguous()
            actual = target.forward(tensor).float().cpu()
            target.synchronize()
            reference_tensor = expected[case_name]
            if actual.shape != reference_tensor.shape:
                case_reports.append(
                    {
                        "case": case_name,
                        "shape_equal": False,
                        "reference_shape": list(reference_tensor.shape),
                        "actual_shape": list(actual.shape),
                        "rtol": tolerance[0],
                        "atol": tolerance[1],
                        "passed": False,
                    }
                )
                continue
            difference = (actual - reference_tensor).abs()
            case_reports.append(
                {
                    "case": case_name,
                    "shape_equal": True,
                    "max_abs_error": float(difference.max()),
                    "mean_abs_error": float(difference.mean()),
                    "cosine_similarity": float(
                        functional.cosine_similarity(actual.flatten(), reference_tensor.flatten(), dim=0)
                    ),
                    "rtol": tolerance[0],
                    "atol": tolerance[1],
                    "passed": bool(torch.allclose(actual, reference_tensor, *tolerance)),
                }
            )
        report[target.name] = {
            "reference": "pytorch_cuda_fp32",
            "passed": all(item["passed"] for item in case_reports),
            "cases": case_reports,
        }
        del target
        gc.collect()
        torch.cuda.empty_cache()

    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "consistency.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return report
