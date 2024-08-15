# [ViCLIP](https://arxiv.org/pdf/2307.06942.pdf): a video-text representation learning model trained on  [InternVid](https://arxiv.org/pdf/2307.06942.pdf)

[![Dataset meta](https://img.shields.io/badge/%F0%9F%A4%97%20InternVid-Dataset-blue)](https://huggingface.co/datasets/OpenGVLab/InternVid) | [![Model Checkpoint](https://img.shields.io/badge/%F0%9F%A4%97%20ViCLIP-Model-purple)](https://huggingface.co/OpenGVLab/ViCLIP)


# :fire: News
- Training code is released.

- InternVid has been accepted for spotlight presentation of ICLR 2024.

- We release a subset [InternVid-Aesthetics-18M](https://huggingface.co/datasets/OpenGVLab/InternVid/viewer/InternVid-10M/AES). It consists of 18 million video clips that have been assigned high aesthetic scores. For more details on the aesthetic scoring, please refer to [laion aesthetic predictor](https://github.com/LAION-AI/aesthetic-predictor).

- We enhance InternVid-10M-FLT dataset annotations by incorporating video language and type information sourced from YouTube's metainfo. You can find the updated annotations at [this link](https://huggingface.co/datasets/OpenGVLab/InternVid-10M-FLT-INFO).

- We release ViCLIP models trained on different subsets of InternVid. Check their performance [here](#model-performance) and download them [here](#pretrained-data--model).

- We are excited to announce the partial release of a large-scale video-text dataset aimed at facilitating multimodal understanding and generation. As part of this release, we are making available a subset [InternVid-10M-FLT](https://huggingface.co/datasets/OpenGVLab/InternVid) of the dataset, which comprises 10 million video clips. Additionally, we have provided a [ViCLIP](https://huggingface.co/OpenGVLab/ViCLIP) model trained on this subset, using the ViT-L architecture. It achieves SOTA zero-shot action recognition performance on Kinetics.

- We give a step-by-step instructions and clarify the process of accessing and utilizing ViClip in [demo.ipynb](https://github.com/OpenGVLab/InternVideo/blob/main/Data/InternVid/demo.ipynb).

- Some model weights and the corresponding data are released at [Pretrained Data & Model](#pretrained-data--model). Their performance is given at [Model Performance](#model-performance).

Stay tuned for updates!

# Introduction

### ViCLIP: a simple video CLIP for transferrable video-text representation

Built upon <a href="https://github.com/openai/CLIP">CLIP</a>, we make a simple video-text pretraining baseline ViCLIP. It consists of a video encoder (ViT) and a text encoder, as given below. Both modules are initialized from the corresponding CLIP components. We update the native attention in the video encoder to spatiotemporal attention while maintaining other design elements. For efficient learning, we apply masking to videos in pre-training.

<img width="633" alt="87c6263cc4aceee72cc8e37085a8109" src="https://github.com/OpenGVLab/InternVideo/assets/43169235/1e540a2b-f503-4036-b2a8-ba99401fc5b0">

### Model Performance

**Table 1: Zero-shot action recognition results on Kinetics 400/600/700. We report the top-1 accuracy of the compared methods on each dataset.**
|Method | Training Data | K400 |  | K600 |  | K700 |  |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| | |top-1 | AVG | top-1 | AVG | top-1 | AVG |
CLIP |	CLIP400M | 58.42 | 70.14 |	55.11|	67.16|	46.12|	58.38
CLIP |	DataComp-1B	|56.14|	67.67|	54.15|	65.83|	45.36|	57.01
EVA-CLIP-L |	Merged-2B |	- |	65.00|	-	|64.90|	-	|59.10
EVA-CLIP-E |	LAION-2B	|-	|69.80|-|	69.30|	-|	63.40
ViCLIP-B | +InternVid-10M-FLT | 58.52 | 71.11 | 55.37 | 68.27 | 47.09 | 59.98 
ViCLIP-B | +InternVid-200M | 56.58 | 69.20 | 53.57 | 66.20 | 45.82 | 58.28 
ViCLIP-L|	+WebVid10M	|59.88|	71.03|	58.66|	69.84|	50.23|	61.86
ViCLIP-L|	+InternVid-10M-DIV|	63.00|	74.15|	60.68|	72.07|	52.50|	64.59
ViCLIP-L|	+InternVid-10M-FLT|	**64.80** |	**75.70** | **62.20** | **73.53** | **54.30** | **66.38**
ViCLIP-L | +InternVid-200M | 59.80 | 71.09 | 57.80 | 69.34 | 49.30 | 61.25 

**Table 2: Fine-tuned action recognition results on Kinetics 400 and SomethingSomethingV2.**
|Method | Training Data | K400 |  | SthSthV2 |  |
|:---:|:---:|:---:|:---:|:---:|:---:|
| | |top-1 | top-5 | top-1 | top-5|
CLIP |	CLIP400M | 86.7 | 97.2 | 70.1 | 92.5
CLIP |	DataComp-1B	|85.6| 96.8| 68.9| 91.8
ViCLIP-L|	+WebVid10M	|85.0| 96.8| 68.7| 91.9
ViCLIP-L|	+InternVid-10M-FLT|	86.8 |97.5| 71.2| 93.2
ViCLIP-L|	+InternVid-10M-FLT+K710|	88.0| 97.8| 71.8| 93.6
ViCLIP-L | +InternVid-200M | 87.9 |97.9| 73.6| 94.9
ViCLIP-L | +InternVid-200M+K710 | **88.7** | **98.2** | **74.2** | **95.0**

# Installation

### Requirements

```
# create
conda env create -f viclip.yaml
# activate
conda activate viclip
```

### Note

To run pretraining, you have to prepare the weights of the CLIP visual encoder as in the [`extract.ipynb`](preprocess/extract_hfclip.ipynb), and set the `MODEL_PATH` in [`clip_vision.py`](models/backbones/clip/clip_vision.py) and [`clip_text.py`](models/backbones/clip/clip_text.py).


# Pre-Training

We use [CLIP](https://github.com/openai/CLIP) and [OpenCLIP](https://github.com/mlfoundations/open_clip) pretrained models as the unmasked teachers by default:
- Follow [extract.ipynb](preprocess/extract_hfclip.ipynb) to extract visual encoder from CLIP.
- Change `MODEL_PATH` in [`clip_vision.py`](models/backbones/clip/clip_vision.py) and [`clip_text.py`](models/backbones/clip/clip_text.py)..

For training, you can simply run the pretraining scripts in `exp/pretraining` as follows:
```shell
bash exp/exp_pretrain_ViCLIP/viclip_base/run.sh
```

:warning: **Notes:**
1. Set `data_dir` and `your_data_path` like `your_webvid_path` in [data.py](./configs/data.py) before running the scripts.
2. Set `vision_encoder.pretrained` in `vision_encoder.pretrained` in the corresponding config files.
3. Set `--rdzv_endpoint` to your `MASTER_NODE:MASTER_PORT`. You can also use the following commond to automatically set it:
    ```shell
    MASTER_NODE=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
    ALL_NODES=$(scontrol show hostnames "$SLURM_JOB_NODELIST")
    MASTER_PORT=$((10000 + $RANDOM % 100))
    torchrun --rdzv_endpoint=${MASTER_NODE}:10068 $@
    ```
4. `save_latest=True` will automatically save the latest checkpoint while training.
5. `auto_resume=True` will automatically loaded the best or latest checkpoint while training.


# Data & Model Zoo

### Pretrained Data & Model
<div>

|      Model      |   Training Data   |                                               Descriptions                                                |
| :-----------------: | :----------------------: | :---------------------------------------------------------------------------------------------------: |
| ViCLIP-L-14 \[[HuggingFace](https://huggingface.co/OpenGVLab/ViCLIP) \| [🤗](https://huggingface.co/OpenGVLab/ViCLIP/resolve/main/ViClip-InternVid-10M-FLT.pth)\] | InternVid-10M-FLT \[[HuggingFace](https://huggingface.co/datasets/OpenGVLab/InternVid) \| [OpenDataLab](https://opendatalab.com/shepshep/InternVid)\] |   - |
| ViCLIP-L-14 \[[Aliyun](https://pjlab-gvm-data.oss-cn-shanghai.aliyuncs.com/internvideo/viclip/ViCLIP-L_InternVid-DIV-10M.pth)\] | InternVid-10M-DIV  |   - |
| ViCLIP-L-14 \[[Aliyun](https://pjlab-gvm-data.oss-cn-shanghai.aliyuncs.com/internvideo/viclip/ViCLIP-L_WebVid-10M.pth)\] | WebVid-10M  |   - |
| ViCLIP-L-14 \[[Aliyun](https://pjlab-gvm-data.oss-cn-shanghai.aliyuncs.com/internvideo/viclip/ViCLIP-L_InternVid-10M.pth)\] | InternVid-10M  |   - |
| ViCLIP-L-14 \[[Aliyun](https://pjlab-gvm-data.oss-cn-shanghai.aliyuncs.com/internvideo/viclip/ViCLIP-L_InternVid-50M.pth)\] | InternVid-50M  |   - |
| ViCLIP-L-14 \[[Aliyun](https://pjlab-gvm-data.oss-cn-shanghai.aliyuncs.com/internvideo/viclip/ViCLIP-L_InternVid-200M.pth)\] | InternVid-200M  |   - |
| ViCLIP-B-16 \[[OneDrive](https://pjlab-my.sharepoint.cn/:u:/g/personal/wangyi_pjlab_org_cn/EY6ac22ZVzJLm1-wm_9gPaMBm5MFg36GKTxlkwTemgmKzQ?e=mH6u6A)\] | InternVid-10M-FLT  |   - |
| ViCLIP-B-16 \[[OneDrive](https://pjlab-my.sharepoint.cn/:u:/g/personal/wangyi_pjlab_org_cn/EVGBg6kq4M1MjbeSdqiXsaMBaBduhR7CQCT11JR4edmZ8Q?e=ILtTfM)\] | InternVid-200M  |   - |
</div>



## Citation

If you find this work useful for your research, please consider citing InternVid. Your acknowledgement would greatly help us in continuing to contribute resources to the research community.

```
@article{wang2023internvid,
  title={InternVid: A Large-scale Video-Text Dataset for Multimodal Understanding and Generation},
  author={Wang, Yi and He, Yinan and Li, Yizhuo and Li, Kunchang and Yu, Jiashuo and Ma, Xin and Chen, Xinyuan and Wang, Yaohui and Luo, Ping and Liu, Ziwei and Wang, Yali and Wang, Limin and Qiao, Yu},
  journal={arXiv preprint arXiv:2307.06942},
  year={2023}
}

@article{wang2022internvideo,
  title={InternVideo: General Video Foundation Models via Generative and Discriminative Learning},
  author={Wang, Yi and Li, Kunchang and Li, Yizhuo and He, Yinan and Huang, Bingkun and Zhao, Zhiyu and Zhang, Hongjie and Xu, Jilan and Liu, Yi and Wang, Zun and Xing, Sen and Chen, Guo and Pan, Junting and Yu, Jiashuo and Wang, Yali and Wang, Limin and Qiao, Yu},
  journal={arXiv preprint arXiv:2212.03191},
  year={2022}
}
```

# Acknowledgement
This repository is built based on [VINDLU](https://github.com/klauscc/VindLU), [UniFormer](https://github.com/Sense-X/UniFormer) and [VideoMAE](https://github.com/MCG-NJU/VideoMAE) repository.

# Discussion Group
If you have any questions during the trial, running or deployment, feel free to join our WeChat group discussion! If you have any ideas or suggestions for the project, you are also welcome to join our WeChat group discussion!

![image](https://github.com/OpenGVLab/Ask-Anything/assets/43169235/c3020408-4d53-490b-8060-7fd54b0ef09c)
