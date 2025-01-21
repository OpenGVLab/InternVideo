# InternVideo2.5 \[[Paper\]]()

<!-- [中文 README](README_cn.md) -->

This repo will give the code and models of '[InternVideo2.5: Empowering Video MLLMS with long and rich context modeling]()'. InternVideo2.5 is a video multimodal large language model (MLLM, built upoon [InternVL2.5](https://github.com/OpenGVLab/InternVL)) enhanced with long and rich context (LRC) modeling. It significantly improves upon existing MLLMs by enhancing their ability to perceive fine-grained details and capture long-form temporal structures. We achieve this through dense vision task annotations using direct preference optimization ([TPO](https://github.com/OpenGVLab/TPO)) and compact spatiotemporal representations via adaptive hierarchical token compression ([HiCo](https://github.com/OpenGVLab/VideoChat-Flash)).

Our experiments demonstrate substantial performance gains on mainstream short and long video understanding benchmarks. InternVideo2.5 can memorize video inputs at least 6x longer than the original model and exhibits specialized vision capabilities like object tracking and segmentation. This work highlights the importance of rich multimodal context (length and detail) for empowering MLLM focus and memory, offering valuable insights for future video MLLM research.

## Updates
- `2025/01/22`: The technical report of InternVideo2.5 is released.
