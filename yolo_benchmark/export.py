from __future__ import annotations

import json
from pathlib import Path

import torch

from .config import Config
from .model import ExportWrapper, load_detection_model


def export_onnx(config: Config) -> Path:
    """在 CPU 上导出固定空间尺寸、动态 batch 的原始 YOLO 输出。"""
    output = config.model.onnx
    output.parent.mkdir(parents=True, exist_ok=True)
    model = ExportWrapper(load_detection_model(config.model.weights, torch.device("cpu"))).eval()
    example = torch.zeros((config.input.batch_size, 3, *config.input.size), dtype=torch.float32)
    with torch.inference_mode():
        torch.onnx.export(
            model,
            example,
            output,
            input_names=["images"],
            output_names=["predictions"],
            opset_version=17,
            do_constant_folding=True,
            dynamo=False,
        )
    import onnx

    graph = onnx.load(str(output))
    onnx.checker.check_model(graph)
    return output


def build_tensorrt(config: Config) -> Path:
    """从 ONNX 构建原生 TensorRT FP16 engine；不提供 CPU 或 FP32 替代路径。"""
    from .backends import require_cuda

    device = require_cuda(config.runtime.device_id)
    if not config.model.onnx.is_file():
        raise FileNotFoundError(f"ONNX 文件不存在，请先执行 export：{config.model.onnx}")
    try:
        import tensorrt as trt
    except ImportError as error:
        raise RuntimeError("未安装 TensorRT Python 包") from error

    # 单独导出固定形状的 FP16 模型，保留原 FP32 ONNX 用于对比。
    import onnx
    from onnxconverter_common import float16

    # 从 FP32 ONNX 转换，保留输入输出及不适合半精度的算子为 FP32。
    fp16_onnx = config.model.onnx.with_name(
        config.model.onnx.stem + "_fp16.onnx"
    )
    shape = (config.input.batch_size, 3, *config.input.size)

    graph = onnx.load(str(config.model.onnx))
    blocked_ops = sorted(set(float16.DEFAULT_OP_BLOCK_LIST) | {"Range"})
    graph = float16.convert_float_to_float16(
        graph,
        keep_io_types=True,
        op_block_list=blocked_ops,
    )
    onnx.checker.check_model(graph)
    onnx.save(graph, str(fp16_onnx))
   
   
    logger = trt.Logger(trt.Logger.INFO)
    builder = trt.Builder(logger)
    network = builder.create_network(0)
    parser = trt.OnnxParser(network, logger)
    if not parser.parse(config.model.onnx.read_bytes()):
        messages = "\n".join(str(parser.get_error(i)) for i in range(parser.num_errors))
        raise RuntimeError(f"TensorRT 解析 ONNX 失败：\n{messages}")
    build_config = builder.create_builder_config()
    build_config.set_memory_pool_limit(
        trt.MemoryPoolType.WORKSPACE, config.runtime.trt_workspace_gib * (1 << 30)
    )
    serialized = builder.build_serialized_network(network, build_config)
    if serialized is None:
        raise RuntimeError("TensorRT FP16 engine 构建失败")

    output = config.model.engine
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_bytes(serialized)
    metadata = {
        "source_onnx": str(fp16_onnx),
        "input_shape": list(shape),
        "precision": "FP16",
        "tensorrt_version": trt.__version__,
        "device": torch.cuda.get_device_name(config.runtime.device_id),
    }
    output.with_suffix(output.suffix + ".json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return output
