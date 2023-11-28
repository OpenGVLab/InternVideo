# InternVid \[[论文](https://arxiv.org/pdf/2307.06942.pdf)\]

[![数据集](https://img.shields.io/badge/%F0%9F%A4%97%20InternVid-Dataset-blue)](https://huggingface.co/datasets/OpenGVLab/InternVid) | [![模型](https://img.shields.io/badge/%F0%9F%A4%97%20ViCLIP-Model-purple)](https://huggingface.co/OpenGVLab/ViCLIP)

\[[English verision](README.md)\]

# :fire: 新闻
我们很高兴宣布部分发布一个大规模的视频文本数据集，旨在促进多模态理解和生成。作为此次发布的一部分，我们提供了该数据集的[子集](https://huggingface.co/datasets/OpenGVLab/InternVid)包含1000万个视频剪辑。此外，我们还提供了一个使用ViT-L架构在这个子集上训练的[ViCLIP](https://huggingface.co/OpenGVLab/ViCLIP)。该模型在Kinetics上实现了SOTA的零样本动作识别性能。

我们提供了示例代码，阐明如何使用ViClip的过程，在[demo.ipynb](https://github.com/OpenGVLab/InternVideo/blob/main/Data/InternVid/demo.ipynb)中有详述。

请关注我们的更新！

# 简介

**数据**

我们从16个流行类别中收集了各种百分比的视频。为了确保多样性，我们选择了来自不同语言的国家的视频，而非依赖于一个主导语言环境。我们采样的国家包括英国、美国、澳大利亚、日本、韩国、中国、俄罗斯和法国等。在时长方面，每个视频平均持续351.9秒。几乎一半（49%）的视频时长不超过五分钟，而四分之一（26%）的视频时长在五到十分钟之间。只有8%的视频超过20分钟。在策划的视频中，85%是高分辨率（720P），其余15%的分辨率从360P至720P不等。虽然低分辨率的视频在内容生成任务中可能表现不如高分辨率的视频，但只要配有适当的字幕，它们仍可用于视频-语言表示学习。

![b469e00b43d46a6b3f89899483abcf6](https://github.com/OpenGVLab/InternVideo/assets/43169235/7d6aca7d-362a-425d-9ef2-ec0189491b52)

InternVid展示了在分割剪辑级别上具有不同剪辑时长和字幕长度的多样性。美学分数和剪辑-字幕相似度均匀分布。大部分剪辑的长度在0-10秒之间，占所有剪辑的85%。大约一半的剪辑字幕含有10-20个单词，而三分之一的剪辑字幕含有少于10个单词。大约11%的剪辑具有超过20个单词的长字幕。

![429af4993adb77478c000c865ae5a1b](https://github.com/OpenGVLab/InternVideo/assets/43169235/f64588c3-81e8-43de-b771-46500474d2ff)

**ViCLIP: 一个简单的用于转移视频-文本表示的视频CLIP**

基于<a href="https://github.com/openai/CLIP">CLIP</a>, 我们构建了一个简单的视频-文本预训练基线ViCLIP。它由视频编码器（ViT）和文本编码器组成，如下所示。这两个模块都是从相应的CLIP组件初始化的。我们将视频编码器中的原生注意力更新为时空注意力，同时保持其他设计元素不变。为了高效学习，我们在预训练中对视频进行了掩蔽处理。

<img width="633" alt="87c6263cc4aceee72cc8e37085a8109" src="https://github.com/OpenGVLab/InternVideo/assets/43169235/1e540a2b-f503-4036-b2a8-ba99401fc5b0">


# 数据 & 模型库

### 预训练数据 & 模型

<div>

|      模型      |   训练数据   |                                               描述                                                |
| :-----------------: | :----------------------: | :---------------------------------------------------------------------------------------------------: |
| ViCLIP-L-14 \[[HuggingFace](https://huggingface.co/OpenGVLab/ViCLIP) \| [Aliyun](https://pjlab-gvm-data.oss-cn-shanghai.aliyuncs.com/internvideo/viclip/ViClip-InternVid-10M-FLT.pth)\] | InternVid-10M-FLT \[[HuggingFace](https://huggingface.co/datasets/OpenGVLab/InternVid) \| [OpenDataLab](https://opendatalab.com/shepshep/InternVid)\] |    |
</div>


## Citation

如果您发现这项工作对您的研究有所帮助，请考虑引用InternVid。您的肯定将极大地帮助我们继续为研究社区贡献资源。

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