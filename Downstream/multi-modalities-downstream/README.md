# Multi-Modalities-Downstream

This is an official implementation of the multi-modalities downstrem tasks in [InternVideo](https://arxiv.org/abs/2212.03191), including zero-shot action recognition, zero-shot multiple choice, and video question answering.

## Usage

### Model preparation

We will update the model soon.

### Installation and data preparation

The code is mostly based on [All-In-One](https://github.com/showlab/all-in-one). Please follow it for installation and data preparation.

### Downstream tasks

#### Zero-shot action recognition

Please follow `scripts/zs_classify.sh`. We provide results on Kinetics-400.

#### Zero-shot multiple choice

Please follow `scripts/zs_choice_[dataset].sh`. We provide results on MSRVTT and LSMDC.

#### Video question answering

Please follow `scripts/finetune_[dataset].sh`. We provide results on MSRVTT, MSVD, and TGIF-FrameQA.

## TODO

The B/16 model is on its way.

## Acknowledgement

This repo is built based on [All-In-One](https://github.com/showlab/all-in-one), [CLIP](https://github.com/openai/CLIP) and [open_clip](https://github.com/mlfoundations/open_clip).
