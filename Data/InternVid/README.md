# InternVid \[[Paper](https://arxiv.org/pdf/2307.06942.pdf)\]

[![Dataset meta](https://img.shields.io/badge/%F0%9F%A4%97%20InternVid-Dataset-blue)](https://huggingface.co/datasets/OpenGVLab/InternVid) | [![Model Checkpoint](https://img.shields.io/badge/%F0%9F%A4%97%20ViCLIP-Model-purple)](https://huggingface.co/OpenGVLab/ViCLIP)

# :fire: News
We are excited to announce the partial release of a large-scale video-text dataset aimed at facilitating multimodal understanding and generation. As part of this release, we are making available a [subset](https://huggingface.co/datasets/OpenGVLab/InternVid) of the dataset, which comprises 10 million video clips. Additionally, we have provided a [ViCLIP](https://huggingface.co/OpenGVLab/ViCLIP) model trained on this subset, using the ViT-L architecture. It achieves SOTA zero-shot action recognition performance on Kinetics.

Stay tuned for updates!

# Introduction

**Data**

We collected videos from 16 popular categories with varying percentages. We ensured diversity by selecting videos from countries with different languages instead of relying on a dominant language environment. The countries we sampled from include the UK, USA, Australia, Japan, Korea, China, Russia, and France, among others. In terms of duration, every video lasts 351.9s on average. Almost half (49%) of the videos are five minutes or less, while a quarter (26%) fall between five and ten minutes. Only 8% of the videos are over 20 minutes long. Among the curated videos, 85% were high-resolution (720P), while the remaining 15% had lower resolutions ranging from 360P to 720P. Although the lower-resolution videos may not perform as well as the high-resolution ones in content generation tasks, they can still be useful in video-language representation learning, provided that they have appropriate captions.

![b469e00b43d46a6b3f89899483abcf6](https://github.com/OpenGVLab/InternVideo/assets/43169235/7d6aca7d-362a-425d-9ef2-ec0189491b52)

InternVid exhibits diverse clip durations and caption lengths in the segmented clip level. The aesthetic scores and clip-caption similarities are distributed uniformly. The majority of clips are 0-10 seconds in length, accounting for 85% of all clips. Approximately half of the clips have captions with 10-20 words, while one-third of the clip captions have fewer than 10 words. About 11% of clips have long captions with more than 20 words.

![429af4993adb77478c000c865ae5a1b](https://github.com/OpenGVLab/InternVideo/assets/43169235/f64588c3-81e8-43de-b771-46500474d2ff)

**ViCLIP: a simple video CLIP for transferrable video-text representation**

Built upon <a href="https://github.com/openai/CLIP">CLIP</a>, we make a simple video-text pretraining baseline ViCLIP. It consists of a video encoder (ViT) and a text encoder, as given below. Both modules are initialized from the corresponding CLIP components. We update the native attention in the video encoder to spatiotemporal attention while maintaining other design elements. For efficient learning, we apply masking to videos in pre-training.

<img width="633" alt="87c6263cc4aceee72cc8e37085a8109" src="https://github.com/OpenGVLab/InternVideo/assets/43169235/1e540a2b-f503-4036-b2a8-ba99401fc5b0">


# Data & Model Zoo

<details>
<summary> Pretrained Data & Model </summary>
<br>
<div>

|      Model      |   Training Data   |                                               Descriptions                                                |
| :-----------------: | :----------------------: | :---------------------------------------------------------------------------------------------------: |
| ViCLIP-L-14 \[[ckpt](https://huggingface.co/OpenGVLab/ViCLIP)\] | InternVid-10M-FLT \[[HuggingFace](https://huggingface.co/datasets/OpenGVLab/InternVid)\] |    |
</div>
</details>

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
