from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Any

from .benchmark import run_benchmark
from .config import Config, load_config
from .consistency import check_consistency
from .environment import collect_environment
from .evaluate import evaluate_coco
from .export import build_tensorrt, export_onnx
from .report import write_report


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="YOLO 目标检测推理性能评测")
    parser.add_argument("--config", default="configs/coco_val.yaml", help="YAML 配置路径")
    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser("export", help="导出 ONNX FP32 模型")
    subparsers.add_parser("build-trt", help="构建 TensorRT FP16 engine")

    check = subparsers.add_parser("check", help="与 PyTorch FP32 检查原始输出一致性")
    _backends(check, default=["onnx"])
    check.add_argument("--random-cases", type=int, default=2)
    check.add_argument("--real-cases", type=int, default=2)
    _output(check)

    benchmark = subparsers.add_parser("benchmark", help="执行三轮性能测试")
    _backends(benchmark)
    _output(benchmark)

    evaluate = subparsers.add_parser("evaluate", help="在 COCO val2017 上评估 bbox mAP")
    _backends(evaluate)
    _output(evaluate)

    run = subparsers.add_parser("run", help="导出、校验、基准测试并评估 mAP")
    _backends(run)
    run.add_argument("--skip-eval", action="store_true", help="跳过 COCO mAP（仅用于快速性能测试）")
    _output(run)
    return parser


def _backends(parser: argparse.ArgumentParser, default: list[str] | None = None) -> None:
    parser.add_argument(
        "--backends",
        nargs="+",
        choices=["torch", "onnx", "trt"],
        default=default or ["torch", "onnx"],
    )


def _output(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--output", help="输出目录；默认 outputs/<时间戳>")


def _output_dir(config: Config, value: str | None) -> Path:
    if value:
        candidate = Path(value)
        return candidate.resolve() if candidate.is_absolute() else (config.root / candidate).resolve()
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return config.runtime.output_dir / stamp


def _write_manifest(config: Config, output_dir: Path, command: str, backends: list[str]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest: dict[str, Any] = {
        "command": command,
        "backends": backends,
        "fp32_math_mode": "IEEE FP32（PyTorch 与 ONNX Runtime CUDA 均关闭 TF32）",
        "config": config.as_dict(),
        "metric_scopes": {
            "inference_ms": "CUDA Event；仅 backend.forward，输入和输出已在 GPU，不含预后处理及传输",
            "end_to_end_ms": "perf_counter；内存中 BGR 图开始，含 letterbox/归一化、H2D、推理、NMS、检测结果 D2H 和 GPU 同步；不含磁盘解码",
            "throughput": "每轮图片数 / 对应累计耗时；未并发、未重叠流水线",
            "gpu_memory": "NVML 当前进程显存；模型加载增量与每批结束采样的观测峰值，不代表瞬时理论峰值",
        },
        "environment": collect_environment(config.runtime.device_id),
    }
    (output_dir / "experiment.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def main() -> None:
    args = _parser().parse_args()
    config = load_config(args.config)

    if args.command == "export":
        print(f"ONNX 已导出：{export_onnx(config)}")
        return
    if args.command == "build-trt":
        print(f"TensorRT engine 已构建：{build_tensorrt(config)}")
        return

    output_dir = _output_dir(config, args.output)
    _write_manifest(config, output_dir, args.command, args.backends)

    if args.command == "check":
        report = check_consistency(config, args.backends, output_dir, args.random_cases, args.real_cases)
        if not all(item["passed"] for item in report.values()):
            raise SystemExit(f"一致性检查失败，详情：{output_dir / 'consistency.json'}")
    elif args.command == "benchmark":
        _, summary = run_benchmark(config, args.backends, output_dir)
        write_report(summary, output_dir)
    elif args.command == "evaluate":
        evaluate_coco(config, args.backends, output_dir)
    elif args.command == "run":
        if any(name in args.backends for name in ("onnx", "trt")) and not config.model.onnx.is_file():
            export_onnx(config)
        if "trt" in args.backends and not config.model.engine.is_file():
            build_tensorrt(config)
        targets = [name for name in args.backends if name != "torch"]
        if targets:
            report = check_consistency(config, targets, output_dir)
            if not all(item["passed"] for item in report.values()):
                raise SystemExit(f"一致性检查失败，停止性能测试：{output_dir / 'consistency.json'}")
        _, summary = run_benchmark(config, args.backends, output_dir)
        write_report(summary, output_dir)
        if not args.skip_eval:
            evaluate_coco(config, args.backends, output_dir)
    print(f"完成，结果目录：{output_dir}")


if __name__ == "__main__":
    main()
