from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path

import numpy as np
import torch

from .config import Config
from .model import load_detection_model, unwrap_prediction


def require_cuda(device_id: int) -> torch.device:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA 不可用：本项目不会把 GPU 后端静默切换到 CPU")
    if device_id >= torch.cuda.device_count():
        raise RuntimeError(f"GPU {device_id} 不存在；当前检测到 {torch.cuda.device_count()} 块 GPU")
    device = torch.device("cuda", device_id)
    torch.cuda.set_device(device)
    # FP32 后端统一关闭 TF32，避免把缩减尾数精度的计算标成 FP32。
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    return device


class Backend(ABC):
    name: str
    precision: str
    input_dtype = torch.float32

    def __init__(self, config: Config) -> None:
        self.config = config
        self.device = require_cuda(config.runtime.device_id)

    @abstractmethod
    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """输入、输出均为当前 GPU 上的 Tensor；调用本身可以是异步的。"""

    def synchronize(self) -> None:
        torch.cuda.synchronize(self.device)


class TorchBackend(Backend):
    name = "pytorch_cuda_fp32"
    precision = "FP32"

    def __init__(self, config: Config) -> None:
        super().__init__(config)
        self.model = load_detection_model(config.model.weights, self.device)

    @torch.inference_mode()
    def forward(self, images: torch.Tensor) -> torch.Tensor:
        if images.device != self.device or images.dtype != torch.float32:
            raise ValueError("PyTorch 后端要求 CUDA FP32 输入")
        return unwrap_prediction(self.model(images))


class OrtCudaBackend(Backend):
    name = "onnxruntime_cuda_fp32"
    precision = "FP32"

    def __init__(self, config: Config) -> None:
        super().__init__(config)
        if not config.model.onnx.is_file():
            raise FileNotFoundError(f"ONNX 文件不存在，请先执行 export：{config.model.onnx}")
        try:
            import onnxruntime as ort
        except ImportError as error:
            raise RuntimeError("未安装 onnxruntime-gpu") from error
        ort.preload_dlls()
        if "CUDAExecutionProvider" not in ort.get_available_providers():
            raise RuntimeError(
                "ONNX Runtime 未提供 CUDAExecutionProvider；请安装 onnxruntime-gpu 并检查 CUDA/cuDNN"
            )

        options = ort.SessionOptions()
        # 输出节点分配日志，检查哪些计算实际运行在 GPU 上。
        options.log_severity_level = 0
        options.log_verbosity_level = 1
        stream = torch.cuda.current_stream(self.device)
        provider_options = {
            "device_id": str(config.runtime.device_id),
            "user_compute_stream": str(stream.cuda_stream),
            "do_copy_in_default_stream": "1",
            "use_tf32": "0",
        }
        try:
            self.session = ort.InferenceSession(
                str(config.model.onnx),
                sess_options=options,
                providers=[("CUDAExecutionProvider", provider_options)],
            )
        except Exception as error:
            raise RuntimeError(f"创建 ONNX Runtime CUDA 会话失败，未回退 CPU：{error}") from error
        if self.session.get_providers()[0] != "CUDAExecutionProvider":
            raise RuntimeError(f"CUDA EP 未生效：{self.session.get_providers()}")

        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name
        self.output_shape = self._resolve_output_shape(self.session.get_outputs()[0].shape)
        self.output = torch.empty(self.output_shape, device=self.device, dtype=torch.float32)

    def _resolve_output_shape(self, shape: list[int | str | None]) -> tuple[int, ...]:
        resolved: list[int] = []
        for index, value in enumerate(shape):
            if isinstance(value, int) and value > 0:
                resolved.append(value)
            elif index == 0:
                resolved.append(self.config.input.batch_size)
            else:
                raise RuntimeError(f"ONNX 输出含未解析的动态维度：{shape}")
        return tuple(resolved)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        if images.device != self.device or images.dtype != torch.float32 or not images.is_contiguous():
            raise ValueError("ONNX Runtime 后端要求连续 CUDA FP32 输入")
        binding = self.session.io_binding()
        binding.bind_input(
            self.input_name,
            "cuda",
            self.config.runtime.device_id,
            np.float32,
            tuple(images.shape),
            images.data_ptr(),
        )
        binding.bind_output(
            self.output_name,
            "cuda",
            self.config.runtime.device_id,
            np.float32,
            self.output_shape,
            self.output.data_ptr(),
        )
        self.session.run_with_iobinding(binding)
        return self.output


class TensorRTBackend(Backend):
    name = "tensorrt_fp16"
    precision = "FP16"

    def __init__(self, config: Config) -> None:
        super().__init__(config)
        if not config.model.engine.is_file():
            raise FileNotFoundError(f"TensorRT engine 不存在，请先执行 build-trt：{config.model.engine}")
        try:
            import tensorrt as trt
        except ImportError as error:
            raise RuntimeError("未安装 TensorRT Python 包") from error

        self.trt = trt
        self.logger = trt.Logger(trt.Logger.WARNING)
        self.runtime = trt.Runtime(self.logger)
        self.engine = self.runtime.deserialize_cuda_engine(config.model.engine.read_bytes())
        if self.engine is None:
            raise RuntimeError("TensorRT engine 反序列化失败；engine 必须在当前 GPU/TensorRT 环境构建")
        self.context = self.engine.create_execution_context()
        self.input_name = next(
            self.engine.get_tensor_name(i)
            for i in range(self.engine.num_io_tensors)
            if self.engine.get_tensor_mode(self.engine.get_tensor_name(i)) == trt.TensorIOMode.INPUT
        )
        self.output_name = next(
            self.engine.get_tensor_name(i)
            for i in range(self.engine.num_io_tensors)
            if self.engine.get_tensor_mode(self.engine.get_tensor_name(i)) == trt.TensorIOMode.OUTPUT
        )
        input_shape = (config.input.batch_size, 3, *config.input.size)
        if not self.context.set_input_shape(self.input_name, input_shape):
            raise RuntimeError(f"TensorRT engine 不接受配置输入形状：{input_shape}")
        output_shape = tuple(self.context.get_tensor_shape(self.output_name))
        if any(value < 0 for value in output_shape):
            raise RuntimeError(f"TensorRT 输出形状未解析：{output_shape}")
        self.input_dtype = _torch_dtype(trt.nptype(self.engine.get_tensor_dtype(self.input_name)))
        self.output_dtype = _torch_dtype(trt.nptype(self.engine.get_tensor_dtype(self.output_name)))
        self.output = torch.empty(output_shape, device=self.device, dtype=self.output_dtype)
        self.context.set_tensor_address(self.output_name, self.output.data_ptr())

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        if images.device != self.device or images.dtype != self.input_dtype or not images.is_contiguous():
            raise ValueError(f"TensorRT 后端要求连续 CUDA {self.input_dtype} 输入")
        self.context.set_tensor_address(self.input_name, images.data_ptr())
        stream = torch.cuda.current_stream(self.device)
        if not self.context.execute_async_v3(stream_handle=stream.cuda_stream):
            raise RuntimeError("TensorRT execute_async_v3 执行失败")
        return self.output


def _torch_dtype(numpy_dtype: type[np.generic]) -> torch.dtype:
    mapping = {np.dtype(np.float32): torch.float32, np.dtype(np.float16): torch.float16}
    try:
        return mapping[np.dtype(numpy_dtype)]
    except KeyError as error:
        raise TypeError(f"不支持的 TensorRT 张量类型：{numpy_dtype}") from error


def create_backend(name: str, config: Config) -> Backend:
    backends = {"torch": TorchBackend, "onnx": OrtCudaBackend, "trt": TensorRTBackend}
    if name not in backends:
        raise ValueError(f"未知后端：{name}，可选值：{', '.join(backends)}")
    return backends[name](config)
