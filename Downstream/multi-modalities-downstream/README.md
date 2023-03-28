# Multi-Modalities-Downstream

This is an official implementation of the multi-modalities downstrem tasks in [InternVideo](https://arxiv.org/abs/2212.03191), including zero-shot action recognition, zero-shot multiple choice, and video question answering.

## Usage

### Pre-trained model preparation

We currently provide the [B/16](https://pjlab-gvm-data.oss-cn-shanghai.aliyuncs.com/internvideo/pretrain/InternVideo-MM-B-16.ckpt) model, please download the model from aliyun. You will also need original CLIP model ViT-B-16. Please modify the /path/to/model in the scripts accordingly.

### Installation and data preparation

The code is mostly based on [All-In-One](https://github.com/showlab/all-in-one). Please follow it for installation and data preparation.

### Downstream tasks

#### Zero-shot action recognition

Please follow `scripts/zs_classify.sh`. We provide results on Kinetics-400 **test set**.

| | B/16 | L/14 |
|:---| ----------- | ----------- |
|K400 | 56.65     | 64.25 |

#### Zero-shot multiple choice

Please follow `scripts/zs_choice_[dataset].sh`. We provide results on MSRVTT and LSMDC.

|  |  B/16 | L/14 |
|:---| ----------- | ----------- |
|MSRVTT| 91.31     | 93.44 |
|LSMDC| 73.96     | 77.26 |

#### Video question answering

Please follow `scripts/finetune_[dataset].sh`. We provide results on MSRVTT, MSVD, and TGIF-FrameQA.

|  |  B/16 | L/14 |
|:---| ----------- | ----------- |
|MSRVTT| 44.58     | 47.14 |
|MSVD| 51.77     | 55.54 |
|TGIF-Frame| 67.83     | 72.22 |

## TODO

The L/14 model is on its way.

## Acknowledgement

This repo is built based on [All-In-One](https://github.com/showlab/all-in-one), [CLIP](https://github.com/openai/CLIP), [CoCa](https://github.com/lucidrains/CoCa-pytorch) and [open_clip](https://github.com/mlfoundations/open_clip).
