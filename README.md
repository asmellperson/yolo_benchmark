# YOLO11 目标检测推理性能评测

使用同一份 Ultralytics YOLO11n COCO 预训练权重，对比 PyTorch CUDA FP32、ONNX Runtime CUDA FP32 和 TensorRT FP16 推理的延迟、吞吐量、显存占用与检测精度。

项目覆盖模型导出、输出一致性检查、性能测试、COCO bbox AP/AR 评估和报告生成，不负责模型训练。权重与数据需另行下载；本文记录一次真实运行的结果，供复现与分析使用。

## 实测结果

测试日期：2026-09-28。设备：NVIDIA A100-SXM4-40GB。模型：YOLO11n，参数量 **2,624,080**，输入 **640 × 640**，batch size **1**。

每个后端预热 20 次，执行 3 轮测试，每轮 200 个 batch。下表汇总每个后端的 600 次测量；这些测量包含重复图片，不代表 600 张独立图片。COCO 精度评估另行使用完整 val2017 的 5000 张图片。

### 延迟与吞吐量

| 后端 | 纯推理均值（ms） | 纯推理 P50（ms） | 纯推理 P95（ms） | 端到端 P50（ms） | 端到端 P95（ms） | 纯推理吞吐量（img/s） | 端到端吞吐量（img/s） |
|---|---:|---:|---:|---:|---:|---:|---:|
| PyTorch CUDA FP32 | 5.751 | 5.859 | 7.858 | 12.061 | 18.099 | 173.90 | 83.83 |
| ONNX Runtime CUDA FP32 | 7.221 | 7.250 | 11.877 | 12.536 | 19.041 | 138.48 | 83.87 |
| TensorRT FP16¹ | 1.820 | 1.523 | 3.217 | 8.137 | 14.317 | 549.45 | 128.52 |

¹ `tensorrt_fp16` 是结果文件中的后端名称。当前构建路径将同源 FP32 ONNX 转换为 FP16 图，保留 FP32 输入/输出和转换器阻止降精度的算子，因此应理解为 **FP16/FP32 混合精度**，不能据名称断言所有层均以 FP16 执行。

本次测量中，TensorRT 相对 PyTorch 的纯推理加速比约为 **3.16×**，端到端吞吐量提高约 **53.3%**。ONNX Runtime 的纯推理均值高于 PyTorch，但两者端到端吞吐量接近。结果只适用于本次软硬件、模型和测试口径，不能推导出所有模型或设备上的后端排名。

![YOLO11n 在 A100 上的推理性能对比](docs/results/a100-yolo11n/comparison.png)

### 分轮结果

| 后端 | 轮次 | 测量 batch 数 | 纯推理均值（ms） | 端到端吞吐量（img/s） |
|---|---|---:|---:|---:|
| PyTorch CUDA FP32 | 1 | 200 | 5.545 | 83.95 |
| PyTorch CUDA FP32 | 2 | 200 | 5.892 | 84.89 |
| PyTorch CUDA FP32 | 3 | 200 | 5.815 | 82.68 |
| ONNX Runtime CUDA FP32 | 1 | 200 | 7.078 | 83.38 |
| ONNX Runtime CUDA FP32 | 2 | 200 | 7.488 | 85.06 |
| ONNX Runtime CUDA FP32 | 3 | 200 | 7.099 | 83.20 |
| TensorRT FP16 | 1 | 200 | 1.861 | 129.38 |
| TensorRT FP16 | 2 | 200 | 1.807 | 128.40 |
| TensorRT FP16 | 3 | 200 | 1.792 | 127.78 |

完整精度的数值保存在 `summary.csv`；本文表格进行了四舍五入。

### 显存

| 后端 | 加载前进程基线（MiB） | 模型加载增量（MiB） | 观测峰值相对基线增量（MiB） |
|---|---:|---:|---:|
| PyTorch CUDA FP32 | 436 | 30 | 222 |
| ONNX Runtime CUDA FP32 | 580 | 68 | 122 |
| TensorRT FP16 | 586 | 200 | 202 |

显存通过 NVML 采样当前进程得到。模型加载增量不等于权重大小，观测峰值也不等于瞬时最大显存。各后端在同一进程中依次运行，CUDA 上下文、库和分配器的保留内存可能影响后续基线，因此这些数据不能当作独立进程下的绝对显存需求。

### COCO val2017 精度

使用全部 5000 张图片，评估类型为 `bbox`。下表保留 COCO 控制台输出的三位小数；目前收录的精度日志为 ONNX Runtime 与 TensorRT，未填入未提供的 PyTorch mAP。

| COCO 指标 | ONNX Runtime CUDA FP32 | TensorRT FP16 |
|---|---:|---:|
| AP@[IoU=0.50:0.95] | 0.387 | 0.387 |
| AP@0.50 | 0.542 | 0.542 |
| AP@0.75 | 0.421 | 0.421 |
| AP small | 0.195 | 0.195 |
| AP medium | 0.424 | 0.424 |
| AP large | 0.561 | 0.561 |
| AR@maxDets=1 | 0.311 | 0.311 |
| AR@maxDets=10 | 0.509 | 0.509 |
| AR@maxDets=100 | 0.560 | 0.560 |
| AR small | 0.332 | 0.332 |
| AR medium | 0.624 | 0.623 |
| AR large | 0.734 | 0.734 |

AP 的 `maxDets=100`；按目标大小划分的 AR 同样使用 `maxDets=100`。项目 NMS 后最多保留 300 个检测，与 COCO 指标中的检测数量上限是不同设置。

两后端的 AP 在展示精度下相同，说明此次混合精度转换未出现明显 AP 下降；这不代表原始预测完全一致，也不代表更高小数精度下指标没有差异。评估进度条的 images/s 不作为纯推理性能指标。

### 原始输出一致性

TensorRT 与 PyTorch CUDA FP32 的一致性检查报告包含 2 个随机输入和 2 张 COCO 图片，四项均被检查程序判定为通过，报告阈值为 `rtol=0.05`、`atol=0.05`。

| 输入 | 输出形状一致 | 最大绝对误差 | 平均绝对误差 | 报告判定 |
|---|---|---:|---:|---|
| random_0 | 是 | 0.827759 | 0.001690 | 通过 |
| random_1 | 是 | 1.757874 | 0.001723 | 通过 |
| coco_0 | 是 | 3.006378 | 0.001830 | 通过 |
| coco_1 | 是 | 1.608521 | 0.002287 | 通过 |

原始输出包含不同数值尺度的预测量，最大绝对误差不能直接解释成最终检测框误差或 mAP 损失。少量输入的一致性检查用于发现明显异常，完整数据集上的 AP/AR 用于评估检测精度。

### 结果来源

| 内容 | 本次运行目录 |
|---|---|
| TensorRT 输出一致性 | `outputs/20260928_135247/` |
| COCO 精度评估 | `outputs/20260928_135852/` |
| 三后端性能测试 | `outputs/20260928_141212/` |

## 评测口径

各后端共用相同的模型来源、图片顺序、输入尺寸、batch size、letterbox、RGB/CHW 转换、归一化、置信度过滤、按类别 NMS、坐标还原和 COCO API。

| 指标 | 测量范围 |
|---|---|
| `inference_ms` | CUDA Event 包围 `backend.forward`；输入和输出位于 GPU，不含预处理、H2D、NMS、D2H |
| `end_to_end_ms` | 从内存中的 BGR 图片开始，包含预处理、H2D、推理、NMS、检测结果 D2H 和 GPU 同步；不含磁盘读取与图片解码 |
| P50 / P95 | 对相应延迟样本计算分位数；总体值合并三轮样本计算，不取三轮分位数的平均 |
| 吞吐量 | 图片数除以对应累计耗时；串行执行，不代表并发服务的最大吞吐量 |
| 显存 | NVML 当前进程显存；记录加载增量及采样得到的峰值增量 |

PyTorch 和 ONNX Runtime FP32 后端显式关闭 TF32。TensorRT 使用同源 ONNX 转换得到的混合精度图。后端依次加载和测量。

本次静态 ONNX 的初始化日志显示：**239 个节点全部分配到 `CUDAExecutionProvider`**。这项结论来自节点分配日志，不能仅凭 `get_available_providers()` 推断，也不保证换用其他模型后所有节点仍在 CUDA 上执行。

## 实测环境

| 项目 | 版本 / 配置 |
|---|---|
| GPU | NVIDIA A100-SXM4-40GB，Compute Capability 8.0 |
| 操作系统 | Linux 5.15.0-187-generic，x86_64，glibc 2.35 |
| NVIDIA 驱动 | 580.173.02 |
| Python | 3.11.15 |
| PyTorch | 2.13.0+cu130 |
| torchvision | 0.28.0 |
| PyTorch CUDA runtime | 13.0 |
| cuDNN | `torch.backends.cudnn.version()` 返回 92000 |
| ONNX | 1.23.0 |
| ONNX Runtime GPU | 1.28.0.dev20260722004，CUDA 13 nightly |
| TensorRT | 11.3.0.99，由 `tensorrt.__version__` 确认 |
| Ultralytics 包版本 | 8.4.101 |
| NumPy | 2.4.6 |
| OpenCV | 5.0.0.93 |
| pycocotools | 2.0.11 |

本次运行使用本地 YOLO-Master 中的 Ultralytics 源码；包版本号不足以唯一标识源码，复现时还应保留对应源码版本。环境元数据中的 `tensorrt: null` 表示包元数据未检出，不能据此判断 TensorRT 未安装。

以上是已运行环境记录，不是所有版本组合的兼容性承诺。Windows 环境未列入本次实测。

## 环境准备

先创建独立 Python 环境，并按设备、CUDA 和 cuDNN 版本安装 PyTorch、ONNX Runtime GPU 与 TensorRT。参考：

- [PyTorch 安装说明](https://pytorch.org/get-started/locally/)
- [ONNX Runtime 安装说明](https://onnxruntime.ai/docs/install/)
- [ONNX Runtime CUDA EP 要求](https://onnxruntime.ai/docs/execution-providers/CUDA-ExecutionProvider.html#requirements)
- [TensorRT 文档](https://docs.nvidia.com/deeplearning/tensorrt/latest/)

本次 CUDA 13 环境使用以下 ORT wheel 来源；nightly 包后续是否仍可获取取决于上游保留策略：

```bash
python -m pip uninstall -y onnxruntime
python -m pip install onnxruntime-gpu==1.28.0.dev20260722004 \
  --index-url https://aiinfra.pkgs.visualstudio.com/PublicPackages/_packaging/ort-cuda-13-nightly/pypi/simple/
```

FP16 ONNX 转换需要 `onnxconverter-common`。安装项目其余依赖时，先核对 `requirements.txt` 与所选环境：若仍保留旧的 ONNX Runtime 1.21–1.26 约束，应同步调整，避免把已安装的 CUDA 13 nightly 替换掉。然后安装项目：

```bash
python -m pip install -r requirements.txt
python -m pip install onnxconverter-common
python -m pip install -e .
```

运行前检查实际导入的环境：

```bash
python - <<'PY'
import torch
import onnxruntime as ort
import tensorrt as trt

print("PyTorch:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
print("CUDA runtime:", torch.version.cuda)
print("cuDNN:", torch.backends.cudnn.version())
print("ORT:", ort.__version__)
print("ORT providers:", ort.get_available_providers())
print("TensorRT:", trt.__version__)
PY
```

不运行 TensorRT 时可省略其安装与检查。CUDA 可用且 provider 列表包含 `CUDAExecutionProvider` 只是前置条件，还需成功创建会话并完成推理。

## 数据与模型

- 模型：[Ultralytics YOLO11n](https://docs.ultralytics.com/models/yolo11/)，配置中的权重名为 `yolo11n.pt`。
- 数据：[COCO 2017](https://cocodataset.org/#download) 的 validation images 和 instances annotations。

下载并解压后，配置应能找到以下内容：

| 路径 | 内容 |
|---|---|
| `data/coco/val2017/` | 5000 张验证集图片 |
| `data/coco/annotations/instances_val2017.json` | 实例标注 |
| `yolo11n.pt` | 官方预训练权重；不存在时由 Ultralytics 下载 |

模型权重、数据集和生成的 TensorRT engine 不需要随源码仓库提交。

## 配置

实验参数位于 `configs/coco_val.yaml`：

```yaml
model:
  weights: yolo11n.pt
  onnx: artifacts/yolo11n_b1_640.onnx
  engine: artifacts/yolo11n_b1_640_fp16.engine

input:
  size: [640, 640]
  batch_size: 1

dataset:
  images: data/coco/val2017
  annotations: data/coco/annotations/instances_val2017.json
  limit: null

postprocess:
  confidence: 0.001
  iou: 0.7
  max_detections: 300

benchmark:
  warmup_iterations: 20
  rounds: 3
  iterations_per_round: 200
  seed: 2026

runtime:
  device_id: 0
  trt_workspace_gib: 4
  output_dir: outputs
```

本次结果使用固定 batch、固定空间尺寸的 ONNX。修改 batch size 或输入尺寸后，应重新导出 ONNX 并构建 TensorRT engine，避免混用旧产物。调试时可减少 `dataset.limit` 或测试迭代数；正式 COCO 评估应使用 `limit: null`。

## 运行

在项目根目录执行以下命令。

### 1. 导出 ONNX

```bash
python -m yolo_benchmark --config configs/coco_val.yaml export
```

### 2. 构建 TensorRT engine（可选）

```bash
python -m yolo_benchmark --config configs/coco_val.yaml build-trt
```

当前 TensorRT 11 构建路径通过 ONNX 类型转换表达混合精度，保留 FP32 I/O，并将 `Range` 加入转换阻止列表；不使用旧版的 `BuilderFlag.FP16` 方式。engine 应在目标运行环境构建，不能假设跨 GPU、TensorRT 版本或输入形状通用。

### 3. 检查输出一致性

```bash
python -m yolo_benchmark --config configs/coco_val.yaml check --backends onnx trt
```

### 4. 测试性能

```bash
python -m yolo_benchmark --config configs/coco_val.yaml benchmark --backends torch onnx trt
```

### 5. 评估 COCO 精度

```bash
python -m yolo_benchmark --config configs/coco_val.yaml evaluate --backends torch onnx trt
```

未安装 TensorRT 时，从上述命令的 `--backends` 中去掉 `trt`。性能测试和 COCO 评估是两项独立工作，分别回答运行速度和检测精度问题。

## 输出文件

各命令默认在 `outputs/YYYYMMDD_HHMMSS/` 下保存对应结果；单次命令不会生成下表中的所有文件。支持 `--output` 的子命令可指定输出目录，具体参数以 `--help` 为准。

| 文件 | 内容 |
|---|---|
| `experiment.json` | 参数、环境及指标口径 |
| `consistency.json` | 原始输出误差、形状、阈值与检查结果 |
| `raw_measurements.csv` | 每轮、每 batch 的计时和显存采样 |
| `summary.csv` | 分轮及总体汇总 |
| `summary.md` | 总体结果 Markdown 表格 |
| `comparison.png` | 性能与显存对比图 |
| `coco_predictions_*.json` | COCO 格式预测 |
| `coco_metrics.json` | COCO bbox AP/AR |

### 将本次结果加入仓库

建议只收录用于复核的报告和图表，不提交整个 `outputs/`。本页图片使用 `docs/results/a100-yolo11n/comparison.png`，在本次实验机器上执行：

```bash
mkdir -p docs/results/a100-yolo11n
cp outputs/20260928_141212/comparison.png docs/results/a100-yolo11n/
cp outputs/20260928_141212/summary.csv docs/results/a100-yolo11n/
cp outputs/20260928_141212/raw_measurements.csv docs/results/a100-yolo11n/
cp outputs/20260928_141212/experiment.json docs/results/a100-yolo11n/benchmark_experiment.json
cp outputs/20260928_135247/consistency.json docs/results/a100-yolo11n/trt_consistency.json
cp outputs/20260928_135852/coco_metrics.json docs/results/a100-yolo11n/
```

发布环境元数据前检查其中的绝对路径，将不需公开的本机路径替换成项目相对路径。重新运行后的目录名不同，应使用对应的新目录，并同步更新本文结果。

## 代码结构

| 模块 | 职责 |
|---|---|
| `yolo_benchmark/cli.py` | 命令行入口与任务调度 |
| `yolo_benchmark/config.py` | YAML 配置读取和路径解析 |
| `yolo_benchmark/model.py` | 模型加载与导出包装 |
| `yolo_benchmark/backends.py` | PyTorch、ONNX Runtime、TensorRT 推理封装 |
| `yolo_benchmark/data.py` | COCO 读取、letterbox、归一化与 batch 组装 |
| `yolo_benchmark/postprocess.py` | 置信度过滤、NMS、坐标还原 |
| `yolo_benchmark/export.py` | ONNX 导出与 TensorRT engine 构建 |
| `yolo_benchmark/consistency.py` | 原始预测一致性检查 |
| `yolo_benchmark/benchmark.py` | 延迟、吞吐量与显存测量 |
| `yolo_benchmark/evaluate.py` | COCO bbox 精度评估 |
| `yolo_benchmark/environment.py` | 环境记录与显存监测 |
| `yolo_benchmark/report.py` | 汇总报告与图表生成 |

## 结果边界

- 测试仅覆盖一张 A100、一个模型、一个输入尺寸和 batch size 1。
- TensorRT 与两个 FP32 后端的对比同时包含执行后端和数值精度的变化，不能把全部收益归因于某一项。
- 当前数据不包含独立进程显存对照或 GPU 负载记录，不能据此排除所有运行环境干扰。
- FP16 一致性检查通过和三位小数 AP 相同均不等于逐元素完全一致。
- 本次记录未包含 PyTorch 的完整 COCO 精度结果，也未在本文列出 ONNX 一致性报告的逐项数值。
