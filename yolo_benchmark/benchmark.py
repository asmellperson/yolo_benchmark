from __future__ import annotations

import gc
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

from .backends import create_backend
from .config import Config
from .data import cycle_batches, load_coco_samples, preprocess_batch, read_image
from .environment import GpuMemoryMonitor
from .postprocess import postprocess


def _percentile(values: list[float], percentile: int) -> float:
    return float(np.percentile(np.asarray(values), percentile))


def _release_cuda_cache() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

def run_benchmark(
    config: Config,
    backend_names: list[str],
    output_dir: Path,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    # 后端创建、数据传输、推理和计时统一使用同一条非默认流。
    device = torch.device("cuda", config.runtime.device_id)
    with torch.cuda.device(device):
        stream = torch.cuda.Stream(device=device)
        stream.wait_stream(torch.cuda.current_stream(device))
        with torch.cuda.stream(stream):
            result = _run_benchmark(config, backend_names, output_dir)
            stream.synchronize()
        return result
def _run_benchmark(config: Config, backend_names: list[str], output_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    samples = load_coco_samples(config.dataset.images, config.dataset.annotations, config.dataset.limit)
    batches = list(
        cycle_batches(
            samples,
            config.input.batch_size,
            config.benchmark.iterations_per_round,
        )
    )
    # 磁盘解码明确排除在端到端延迟之外；只缓存实际会测量的图片。
    unique = {sample.image_id: sample for batch in batches for sample in batch}
    decoded = {image_id: read_image(sample) for image_id, sample in unique.items()}
    raw_rows: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []

    for backend_key in backend_names:
        monitor = GpuMemoryMonitor(config.runtime.device_id)
        baseline_mib = monitor.current_mib()
        backend = create_backend(backend_key, config)
        loaded_mib = monitor.current_mib()

        first = batches[0]
        first_cpu, first_metas = preprocess_batch(first, [decoded[x.image_id] for x in first], config.input.size)
        first_gpu = first_cpu.to(backend.device, dtype=backend.input_dtype).contiguous()
        for _ in range(config.benchmark.warmup_iterations):
            prediction = backend.forward(first_gpu)
            postprocess(
                prediction,
                first_metas,
                config.postprocess.confidence,
                config.postprocess.iou,
                config.postprocess.max_detections,
            )
        backend.synchronize()
        monitor.current_mib()

        backend_rows: list[dict[str, Any]] = []
        for round_index in range(1, config.benchmark.rounds + 1):
            inference_times: list[float] = []
            for iteration, batch in enumerate(batches, start=1):
                cpu_tensor, _ = preprocess_batch(
                    batch, [decoded[x.image_id] for x in batch], config.input.size
                )
                gpu_tensor = cpu_tensor.to(backend.device, dtype=backend.input_dtype).contiguous()
                backend.synchronize()
                start_event = torch.cuda.Event(enable_timing=True)
                end_event = torch.cuda.Event(enable_timing=True)
                start_event.record()
                backend.forward(gpu_tensor)
                end_event.record()
                end_event.synchronize()
                inference_times.append(float(start_event.elapsed_time(end_event)))
                monitor.current_mib()

            end_to_end_times: list[float] = []
            for iteration, batch in enumerate(batches, start=1):
                started = time.perf_counter_ns()
                cpu_tensor, metas = preprocess_batch(
                    batch, [decoded[x.image_id] for x in batch], config.input.size
                )
                gpu_tensor = cpu_tensor.to(backend.device, dtype=backend.input_dtype).contiguous()
                prediction = backend.forward(gpu_tensor)
                detections = postprocess(
                    prediction,
                    metas,
                    config.postprocess.confidence,
                    config.postprocess.iou,
                    config.postprocess.max_detections,
                )
                _ = [item.cpu() for item in detections]
                backend.synchronize()
                elapsed_ms = (time.perf_counter_ns() - started) / 1e6
                end_to_end_times.append(elapsed_ms)
                current_mib = monitor.current_mib()
                row = {
                    "backend": backend.name,
                    "precision": backend.precision,
                    "round": round_index,
                    "iteration": iteration,
                    "batch_size": config.input.batch_size,
                    "inference_ms": inference_times[iteration - 1],
                    "end_to_end_ms": elapsed_ms,
                    "gpu_memory_process_mib": current_mib,
                }
                raw_rows.append(row)
                backend_rows.append(row)

            images_count = len(batches) * config.input.batch_size
            summary_rows.append(
                _summary_row(
                    backend.name,
                    backend.precision,
                    f"round_{round_index}",
                    inference_times,
                    end_to_end_times,
                    images_count,
                    baseline_mib,
                    loaded_mib,
                    monitor.peak_mib,
                )
            )

        summary_rows.append(
            _summary_row(
                backend.name,
                backend.precision,
                "overall",
                [row["inference_ms"] for row in backend_rows],
                [row["end_to_end_ms"] for row in backend_rows],
                len(backend_rows) * config.input.batch_size,
                baseline_mib,
                loaded_mib,
                monitor.peak_mib,
            )
        )
        monitor.close()
        del prediction, detections, first_gpu, gpu_tensor, _
        del backend
        _release_cuda_cache()

    raw = pd.DataFrame(raw_rows)
    summary = pd.DataFrame(summary_rows)
    output_dir.mkdir(parents=True, exist_ok=True)
    raw.to_csv(output_dir / "raw_measurements.csv", index=False, encoding="utf-8-sig")
    summary.to_csv(output_dir / "summary.csv", index=False, encoding="utf-8-sig")
    return raw, summary


def _summary_row(
    backend: str,
    precision: str,
    scope: str,
    inference: list[float],
    end_to_end: list[float],
    images_count: int,
    baseline_mib: float,
    loaded_mib: float,
    peak_mib: float,
) -> dict[str, Any]:
    return {
        "backend": backend,
        "precision": precision,
        "scope": scope,
        "samples": images_count,
        "inference_mean_ms": float(np.mean(inference)),
        "inference_p50_ms": _percentile(inference, 50),
        "inference_p95_ms": _percentile(inference, 95),
        "end_to_end_p50_ms": _percentile(end_to_end, 50),
        "end_to_end_p95_ms": _percentile(end_to_end, 95),
        "inference_throughput_img_s": images_count / (sum(inference) / 1000),
        "end_to_end_throughput_img_s": images_count / (sum(end_to_end) / 1000),
        "gpu_memory_baseline_mib": baseline_mib,
        "gpu_memory_model_loaded_mib": max(0.0, loaded_mib - baseline_mib),
        "gpu_memory_peak_delta_mib": max(0.0, peak_mib - baseline_mib),
    }
