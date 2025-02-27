# InternVideo2.5 \[[Paper\]](https://arxiv.org/pdf/2501.12386)

<!-- [中文 README](README_cn.md) -->

This repo will give the code and models of '[InternVideo2.5: Empowering Video MLLMS with long and rich context modeling](https://arxiv.org/pdf/2501.12386)'. InternVideo2.5 is a video multimodal large language model (MLLM, built upon [InternVL2.5](https://github.com/OpenGVLab/InternVL)) enhanced with long and rich context (LRC) modeling. It significantly improves upon existing MLLMs by enhancing their ability to perceive fine-grained details and capture long-form temporal structures. We achieve this through dense vision task annotations using direct preference optimization ([TPO](https://github.com/OpenGVLab/TPO)) and compact spatiotemporal representations via adaptive hierarchical token compression ([HiCo](https://github.com/OpenGVLab/VideoChat-Flash)).

Our experiments demonstrate substantial performance gains on mainstream short and long video understanding benchmarks. InternVideo2.5 can memorize video inputs at least 6x longer than the original model and exhibits specialized vision capabilities like object tracking and segmentation. This work highlights the importance of rich multimodal context (length and detail) for empowering MLLM focus and memory, offering valuable insights for future video MLLM research.

## Updates
- `2025/01/23`: [InternVideo2.5 (InternVL2.5 + LRC)](https://huggingface.co/OpenGVLab/InternVideo2_5_Chat_8B) and [InternVL2.5-HiCo](https://huggingface.co/OpenGVLab/InternVL_2_5_HiCo_R16) have been officially released on HuggingFace.
- `2025/01/22`: The [technical report](https://arxiv.org/pdf/2501.12386) of InternVideo2.5 is released.

https://github.com/user-attachments/assets/1672a633-b28e-4223-acfe-bb1f9a9aa869

https://github.com/user-attachments/assets/6658f32e-ee37-4368-a650-7a0710dcfdd1

https://github.com/user-attachments/assets/db31645a-f405-4545-8367-150cda427981

https://github.com/user-attachments/assets/4478e5df-cb8c-4ec2-989c-944a9d00b4b1

## Model Zoo
| MLLM | Link |  MVBench | Perception Test | LongVideoBench | MLVU | VideoMME | LVBench | #Tokens per frame | #Params |
| ---  | ---  | --- | --- | --- | --- | --- | --- | --- | --- |
| InternVideo2.5 | [huggingface](https://huggingface.co/OpenGVLab/InternVideo2_5_Chat_8B)| 75.7 | 74.9 | 60.6 | 72.8 | 65.1 | 46.4 | 16 | 8B |
| InternVL2.5 + HiCo | [huggingface](https://huggingface.co/OpenGVLab/InternVL_2_5_HiCo_R16) | 74.0 | 71.4 | 59.6 | 71.5 | 64.9 | - | 16 | 8B |
| InternVL2.5 + HiCo | [huggingface](https://huggingface.co/OpenGVLab/InternVL_2_5_HiCo_R64) | 74.4 | 71.9 | 62.7 | 72.6 | 66.4 | - | 64 | 8B |

## Training

See [Finetuning Code](https://github.com/OpenGVLab/VideoChat-Flash/tree/main/xtuner-train_internvideo2_5).
## Citation

If this work is helpful for your research, please consider citing InternVideo2.5.

```
@article{wang2025internvideo,
  title={InternVideo2.5: Empowering Video MLLMs with Long and Rich Context Modeling},
  author={Wang, Yi and Li, Xinhao and Yan, Ziang and He, Yinan and Yu, Jiashuo and Zeng, Xiangyu and Wang, Chenting and Ma, Changlian and Huang, Haian and Gao, Jianfei and Dou, Min and Chen, Kai and Wang, Wenhai and Qiao, Yu and Wang, Yali and Wang, Limin},
  journal={arXiv preprint arXiv:2501.12386},
  year={2025}
}
```
