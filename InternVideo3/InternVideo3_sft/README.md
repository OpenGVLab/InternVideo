# InternVideo3 SFT Training

基于 [XTuner](https://github.com/InternLM/xtuner) 框架的 InternVideo3 监督微调 (SFT) 训练代码。

## 目录结构

```
InternVideo3_sft/
├── xtuner/                  # XTuner 训练框架核心代码
│   └── v1/                  # V1 API
│       ├── model/compose/internvideo3/  # InternVideo3 模型定义
│       ├── datasets/        # 数据集加载与处理
│       ├── train/           # 训练器 (Trainer)
│       ├── loss/            # 损失函数
│       ├── config/          # 配置类 (FSDP, Optimizer, LR)
│       └── engine/          # 训练引擎
├── configs/                 # 训练配置文件
│   ├── internvideo3_sft.py       # 完整 SFT 配置 (64 GPU, batch_size=128)
│   └── internvideo3_sft_debug.py # 调试配置 (单 GPU, batch_size=2)
├── scripts/                 # 启动脚本
│   ├── install.sh           # 安装依赖
│   ├── train.sh             # 训练入口脚本
│   ├── train_debug.sh       # 单卡调试
│   ├── rjob_submit.sh       # rjob 集群提交
│   └── cluster_entrypoint.sh # 集群节点入口
├── pyproject.toml           # Python 包配置
└── README.md
```

## 快速开始

### 1. 安装

```bash
cd InternVideo3_sft
bash scripts/install.sh
```

### 2. 单卡调试

```bash
bash scripts/train_debug.sh
```

### 3. 多卡训练 (单节点)

```bash
bash scripts/train.sh configs/internvideo3_sft.py 8
```

### 4. 集群提交 (rjob)

```bash
# 64 GPU (8 nodes x 8 GPUs)
bash scripts/rjob_submit.sh 64 configs/internvideo3_sft.py

# 自定义 GPU 数量
bash scripts/rjob_submit.sh 32 configs/internvideo3_sft.py
```

## 配置说明

### 环境变量

训练配置支持通过环境变量覆盖关键路径:

| 环境变量 | 说明 | 默认值 |
|---------|------|--------|
| `META_DATA_PATH` | 数据集 meta JSON 路径 | `.../internvideo3_metas/jtx.json` |
| `WORK_DIR` | 检查点保存目录 | `.../internvideo3_sft` |
| `LOAD_FROM` | 预训练模型路径 | `.../hf-1100` |
| `PROCESSOR_PATH` | Tokenizer/Processor 路径 | `.../Qwen3-VL-8B-Instruct` |
| `GLOBAL_BATCH_SIZE` | 全局 batch size | `128` |
| `CEPH_CONFIG` | Ceph/OSS 配置文件 | `.../petreloss.conf` |
| `TOKENIZER_CACHE_DIR` | Tokenizer 缓存目录 | - |

### 模型架构

- **Vision Encoder**: InternVideoNext
- **Projector**: MLA (Multi-head Latent Attention)
- **Language Model**: Qwen3-8B

### 训练超参数

| 参数 | 值 |
|------|-----|
| Learning Rate | 4e-5 |
| LR Schedule | Cosine (min=1e-6) |
| Warmup Ratio | 0.1 |
| Weight Decay | 0.05 |
| Max Sequence Length | 32768 |
| Loss | CE (chunk mode, square reduction) |
| FSDP | CPU offload, recompute_ratio=1.0 |
| Video FPS | 2 |

### 数据格式

Meta JSON 文件格式 (`META_DATA_PATH`):

```json
{
    "dataset_name": {
        "annotation": "/path/to/annotation.jsonl",
        "media_root": "/path/to/media/",
        "sample_ratio": 1.0
    }
}
```

## 依赖

- Python >= 3.10
- PyTorch >= 2.6.0
- transformers == 4.57.3
- Flash Attention 3
- FSDP (Fully Sharded Data Parallel)
