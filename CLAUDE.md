# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

InternVideo is a collection of video foundation models for multimodal understanding, developed by OpenGVLab (Shanghai AI Lab). It contains four major generations plus associated datasets, each in its own top-level directory:

- **InternVideo1/** — Original (2022): generative + discriminative video foundation models (VideoMAE, ViCLIP, UniFormerV2)
- **InternVideo2/** — Scaled (2024): two branches — `single_modality/` (pure video) and `multi_modality/` (video-text)
- **InternVideo2.5/** — Video MLLMs with long/rich context (LRC), built on InternVL2.5
- **InternVideo-Next/** — Latest (2025): video foundation models without video-text supervision
- **Data/** — InternVid (230M video-text pairs) and instruction data (11K samples)

Each generation is largely self-contained with its own models, configs, scripts, and requirements.

## Key Dependencies

All sub-projects share a common deep learning stack:
- Python 3.8+ (3.10+ for multi_modality pyproject.toml install)
- PyTorch (1.13.1+cu117 for older scripts; 2.4.1+ for pyproject.toml)
- NVIDIA apex, DeepSpeed, Flash Attention 2 (with CUDA extensions: fused_dense_lib, layer_norm)
- Video decoding: `decord`
- Vision: `timm`, `einops`, `opencv-python`
- Audio (multi_modality only): `librosa`, `soundfile`, `torchaudio`

## Installation

### InternVideo2 Single Modality
```bash
cd InternVideo2/single_modality
pip install -r requirements.txt
```
Requires pre-downloaded [InternVL-6B visual encoder](https://huggingface.co/OpenGVLab/InternVL/blob/main/internvl_c_13b_224px.pth) and [VideoMAEv2-g](https://github.com/OpenGVLab/VideoMAEv2/blob/master/docs/MODEL_ZOO.md) weights. Set paths in `models/internvl_clip_vision.py` and `models/videomae.py`.

### InternVideo2 Multi Modality
```bash
cd InternVideo2/multi_modality
pip install -r requirements.txt
# Or via pyproject.toml (includes CUDA extension deps):
MAX_JOBS=24 pip install ".[extra-git-deps]"
```
Additionally install FlashAttention2 CUDA extensions (fused_dense_lib and layer_norm) from the flash-attention repo's `csrc/` directories. Set InternVL-6B path in `models/backbones/internvideo2/internvl_clip_vision.py`.

## Training and Evaluation Commands

All training scripts use SLURM (`srun`) for multi-node multi-GPU distributed training. Adjust `PARTITION`, `GPUS`, `GPUS_PER_NODE` for your cluster.

### InternVideo2 Single Modality (from `InternVideo2/single_modality/`)

Entry points: `run_pretraining.py`, `run_finetuning.py`, `run_linear_probing.py`, `run_distill.py`

```bash
# Pretraining (1B model, 128 GPUs)
bash scripts/pretraining/1B_pt.sh

# Finetuning on K400 (32 GPUs)
INTERNVIDEO2_DATA_PATH=/path/to/data INTERNVIDEO2_MODEL_PATH=/path/to/models \
  bash scripts/finetuning/full_tuning/k400/1B_ft_k710_ft_k400_f8.sh

# Distillation
bash scripts/distillation/B14_dist_1B_stage2.sh
```

### InternVideo2 Multi Modality (from `InternVideo2/multi_modality/`)

```bash
# Pretraining (stage2, 1B)
bash scripts/pretraining/stage2/1B/run.sh

# CLIP pretraining
bash scripts/pretraining/clip/1B/run.sh

# Zero-shot evaluation (e.g., MSR-VTT with 1B CLIP model)
bash scripts/evaluation/clip/zero_shot/1B/eval_msrvtt.sh

# Stage2 zero-shot evaluation
bash scripts/evaluation/stage2/zero_shot/1B/eval_msrvtt.sh
```

Evaluation configs are Python files paired with shell scripts under `scripts/evaluation/`. Each config defines dataset paths, model architecture, and evaluation parameters.

## Architecture Notes

### InternVideo2 Single Modality
- Vision Transformer backbone with masked video modeling (VideoMAE-style)
- Uses InternVL-6B as a CLIP teacher for distillation during pretraining
- Model variants: S14, B14, L14, 1B, 6B (patch size in name)
- Config is passed entirely via command-line args to `run_*.py` scripts

### InternVideo2 Multi Modality
- Dual-encoder architecture: video encoder (InternVideo2 backbone) + text encoder (BERT or InternVL)
- Audio branch (BEATs backbone) for audio-visual tasks
- Two training stages: CLIP-style contrastive (stage1/clip) then video-text matching (stage2)
- Config system: Python config files in `scripts/` directories that define `Config` dicts, loaded at runtime
- Model backbones under `models/backbones/` (internvideo2, bert, beats)

### InternVideo-Next
- Minimal codebase in `models/` — `InternVideo_next.py` is the core architecture
- Uses CrossAttention, AttentiveBlock with Flash Attention
- 3D sincos positional embeddings, FusedMLP, DropoutAddRMSNorm from flash_attn

### InternVideo1
- Multiple independent sub-projects under `Pretrain/` and `Downstream/`
- `Pretrain/`: VideoMAE, Multi-Modalities-Pretraining, ViCLIP, UniFormerV2 (git submodule)
- `Downstream/`: action recognition, temporal localization, video retrieval, VQA, open-set recognition

## Git Submodules

Two submodules (init with `git submodule update --init`):
- `InternVideo1/Pretrain/UniFormerV2` → github.com/OpenGVLab/UniFormerV2
- `InternVideo1/Downstream/Ego-Tasks` → github.com/OpenGVLab/ego4d-eccv2022-solutions

## Environment Variables

- `INTERNVIDEO2_DATA_PATH` — root path to datasets (used in finetuning/eval scripts)
- `INTERNVIDEO2_MODEL_PATH` — root path to pretrained model weights
- `MASTER_PORT` — auto-set in scripts via `$((12000 + $RANDOM % 20000))`
- `OMP_NUM_THREADS=1` — standard for distributed training scripts
