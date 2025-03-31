# Model Inference Benchmark

本项目提供了两种模型推理实现：ONNX Runtime 和 OpenCV。基准测试表明，在CPU环境下ONNX Runtime的推理速度更快。

## 项目结构
.
├── config/                  # 配置文件目录
│   ├── config.yaml          # 示例YAML配置文件
│   └── ...                  # 其他模型/任务相关配置
│
├── onnx_model/              # ONNX模型存储目录
│   ├── model1.onnx          # 示例ONNX模型文件
│   └── ...                  
│
├── onnx.py                 # ONNX Runtime推理脚本
├── opencv.py               # OpenCV推理脚本
│
├── README.md               # 项目说明文档
└── requirements.txt        # 可选：依赖库列表
## 环境要求

### ONNX Runtime推理 (onnx.py)
- argparse
- cv2 (OpenCV)
- numpy
- onnxruntime
- yaml
- time (Python标准库)

### OpenCV推理 (opencv.py)
- argparse
- time (Python标准库)
- cv2 (OpenCV)
- numpy
- yaml
## 性能说明
经实际测试，**在CPU环境下ONNX Runtime的推理速度优于OpenCV实现**。

## 使用方法

### ONNX Runtime推理
```bash
python onnx.py --model 模型路径 --yaml 模型路径  --source "0"
python opencv.py --model 模型路径 --yaml 模型路径 --source "0"
