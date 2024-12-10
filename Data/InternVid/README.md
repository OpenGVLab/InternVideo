# InternVid: A Large-scale Video-Text Dataset for Multimodal Understanding and Generation \[[Paper](https://arxiv.org/pdf/2307.06942.pdf)\]

[![Dataset meta](https://img.shields.io/badge/%F0%9F%A4%97%20InternVid-Dataset-blue)](https://huggingface.co/datasets/OpenGVLab/InternVid) | [![Model Checkpoint](https://img.shields.io/badge/%F0%9F%A4%97%20ViCLIP-Model-purple)](https://huggingface.co/OpenGVLab/ViCLIP)

\[[中文版本](README_CN.md)\]

# :fire: News
- We release the search [queries](./queries.jsonl) used in YouTube to retrieve partial video sources.

- We release the full version of the video annotation(230M video-text pairs) for InternVid ([OpenDataLab](https://opendatalab.com/shepshep/InternVidFull) | [HuggingFace](https://huggingface.co/datasets/OpenGVLab/InternVid-Full)). The corresponding clip-text similarities and aesthetics will be updated.

- The implementation of ViCLIP is given [here](https://github.com/OpenGVLab/InternVideo/tree/main/InternVideo1/Pretrain/ViCLIP).

- InternVid has been accepted for spotlight presentation of ICLR 2024.

- We release a subset [InternVid-Aesthetics-18M](https://huggingface.co/datasets/OpenGVLab/InternVid/viewer/InternVid-10M/AES). It consists of 18 million video clips that have been assigned high aesthetic scores. For more details on the aesthetic scoring, please refer to [laion aesthetic predictor](https://github.com/LAION-AI/aesthetic-predictor).

- We enhance InternVid-10M-FLT dataset annotations by incorporating video language and type information sourced from YouTube's metainfo. You can find the updated annotations at [this link](https://huggingface.co/datasets/OpenGVLab/InternVid-10M-FLT-INFO).

- We release ViCLIP models trained on different subsets of InternVid. Check their performance [here](#model-performance) and download them [here](#pretrained-data--model).

- We are excited to announce the partial release of a large-scale video-text dataset aimed at facilitating multimodal understanding and generation. As part of this release, we are making available a subset [InternVid-10M-FLT](https://huggingface.co/datasets/OpenGVLab/InternVid) of the dataset, which comprises 10 million video clips. Additionally, we have provided a [ViCLIP](https://huggingface.co/OpenGVLab/ViCLIP) model trained on this subset, using the ViT-L architecture. It achieves SOTA zero-shot action recognition performance on Kinetics.

- We give a step-by-step instructions and clarify the process of accessing and utilizing ViClip in [demo.ipynb](https://github.com/OpenGVLab/InternVideo/blob/main/Data/InternVid/demo.ipynb).

- Some model weights and the corresponding data are released at [Pretrained Data & Model](#pretrained-data--model). Their performance is given at [Model Performance](#model-performance).

Stay tuned for updates!

# Introduction

### Data

We collected videos from 16 popular categories with varying percentages. We ensured diversity by selecting videos from countries with different languages instead of relying on a dominant language environment. The countries we sampled from include the UK, USA, Australia, Japan, Korea, China, Russia, and France, among others. In terms of duration, every video lasts 351.9s on average. Almost half (49%) of the videos are five minutes or less, while a quarter (26%) fall between five and ten minutes. Only 8% of the videos are over 20 minutes long. Among the curated videos, 85% were high-resolution (720P), while the remaining 15% had lower resolutions ranging from 360P to 720P. Although the lower-resolution videos may not perform as well as the high-resolution ones in content generation tasks, they can still be useful in video-language representation learning, provided that they have appropriate captions.

![b469e00b43d46a6b3f89899483abcf6](https://github.com/OpenGVLab/InternVideo/assets/43169235/7d6aca7d-362a-425d-9ef2-ec0189491b52)

InternVid exhibits diverse clip durations and caption lengths in the segmented clip level. The aesthetic scores and clip-caption similarities are distributed uniformly. The majority of clips are 0-10 seconds in length, accounting for 85% of all clips. Approximately half of the clips have captions with 10-20 words, while one-third of the clip captions have fewer than 10 words. About 11% of clips have long captions with more than 20 words.

![429af4993adb77478c000c865ae5a1b](https://github.com/OpenGVLab/InternVideo/assets/43169235/f64588c3-81e8-43de-b771-46500474d2ff)

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

# Data & Model Zoo

### Pretrained Data & Model
<div>

|      Model      |   Training Data   |                                               Descriptions                                                |
| :-----------------: | :----------------------: | :---------------------------------------------------------------------------------------------------: |
| ViCLIP-L-14 \[[HuggingFace](https://huggingface.co/OpenGVLab/ViCLIP-L-14-hf)\] | InternVid-10M-FLT \[[HuggingFace](https://huggingface.co/datasets/OpenGVLab/InternVid) \| [OpenDataLab](https://opendatalab.com/shepshep/InternVid)\] |   - |
| ViCLIP-L-14  | InternVid-10M-DIV  |   - |
| ViCLIP-L-14  | WebVid-10M  |   - |
| ViCLIP-L-14 | InternVid-10M  |   - |
| ViCLIP-L-14 | InternVid-50M  |   - |
| ViCLIP-L-14 | InternVid-200M  |   - |
| ViCLIP-B-16 \[[OneDrive](https://pjlab-my.sharepoint.cn/:u:/g/personal/wangyi_pjlab_org_cn/EY6ac22ZVzJLm1-wm_9gPaMBm5MFg36GKTxlkwTemgmKzQ?e=mH6u6A)\] \| \[[HuggingFace](https://huggingface.co/OpenGVLab/ViCLIP-B-16-hf)\] | InternVid-10M-FLT  |   - |
| ViCLIP-B-16 \[[OneDrive](https://pjlab-my.sharepoint.cn/:u:/g/personal/wangyi_pjlab_org_cn/EVGBg6kq4M1MjbeSdqiXsaMBaBduhR7CQCT11JR4edmZ8Q?e=ILtTfM)\] | InternVid-200M  |   - |
</div>


## Citation

If you find this work useful for your research, please consider citing InternVid. Your acknowledgement would greatly help us in continuing to contribute resources to the research community.

```
@inproceedings{wang2023internvid,
  title={InternVid: A Large-scale Video-Text Dataset for Multimodal Understanding and Generation},
  author={Wang, Yi and He, Yinan and Li, Yizhuo and Li, Kunchang and Yu, Jiashuo and Ma, Xin and Li, Xinhao and Chen, Guo and Chen, Xinyuan and Wang, Yaohui and others},
  booktitle={The Twelfth International Conference on Learning Representations},
  year={2023}
}

@article{wang2022internvideo,
  title={InternVideo: General Video Foundation Models via Generative and Discriminative Learning},
  author={Wang, Yi and Li, Kunchang and Li, Yizhuo and He, Yinan and Huang, Bingkun and Zhao, Zhiyu and Zhang, Hongjie and Xu, Jilan and Liu, Yi and Wang, Zun and Xing, Sen and Chen, Guo and Pan, Junting and Yu, Jiashuo and Wang, Yali and Wang, Limin and Qiao, Yu},
  journal={arXiv preprint arXiv:2212.03191},
  year={2022}
}
```
