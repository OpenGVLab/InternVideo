# Multi-Modalities-Pretraining

This is an official implementation of the multi-modalities pre-training model in 
[InternVideo](https://arxiv.org/abs/2212.03191), which is resposible for multi-modalities tasks including zero-shot action recognition, zero-shot multiple choice, zero-shot retrieval, video question answering, video-text retrieval, and also one of the component in the final InternVideo model.

## Usage

### Model preparation

We currently provide the [B/16](https://pjlab-gvm-data.oss-cn-shanghai.aliyuncs.com/internvideo/pretrain/InternVideo-MM-B-16.ckpt) model, please download the model from aliyun and place them under folder `models`. The model uses [UniformerV2](https://github.com/OpenGVLab/UniFormerV2) as backbone, and is trained for 12 days using 128 NVIDIA A100 GPUs.

### Demo

To classify the demo video of an airplane taking off, run `python demo.py`, and hopefully you'll see the results as (L/14 model)

```
Label probs:
an airplane is taking off     : 0.9562
an airplane is flying         : 0.0438
a dog is chasing a ball       : 0.0000
```

### For downstream tasks

This folder aims at providing a minimal inference implementation for easier usage. For training and fine-tuning for downstream tasks, please refer to other specific folders.

If you intend to use InternVideo for your own video-language tasks, use the video encoder and text encoder only for alignment tasks such as retrieval, and use the features from cross-modality decoder at the same time if your task involves modalities fusion such as video question answering.

## TODO

The training code and L/14 model is on its way.

## Acknowledgement

This repo is built based on [CLIP](https://github.com/openai/CLIP) and [open_clip](https://github.com/mlfoundations/open_clip).
