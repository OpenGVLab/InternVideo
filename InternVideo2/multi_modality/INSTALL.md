# Installation

## Requirements

We mainly follow [UMT](https://github.com/OpenGVLab/Unmasked_Teacher) to prepare the enviroment.

```shell
pip install -r requirements.txt
```
In addition, in order to support the InternVideo2-6B pre-training, you also need to install [Flash Attention](https://github.com/Dao-AILab/flash-attention) and [DeepSpeed](https://github.com/microsoft/DeepSpeed).


## Note

To run InternVideo2 pretraining, you have to prepare the weights of the **[InternVL-6B visual encoder](https://huggingface.co/OpenGVLab/InternVL/blob/main/internvl_c_13b_224px.pth)**, and set the `your_model_path` in [internvl_clip_vision.py](./models/backbones/internvideo2/internvl_clip_vision.py).

## Key Dependencies Installation for FlashAttention2

Some modules (FusedMLP and DropoutLayerNorm) from FlashAttention2 used in our models rely on CUDA extensions. 

1. Prerequisite for installation: Refer to the [requirements](https://github.com/Dao-AILab/flash-attention?tab=readme-ov-file#installation-and-features) in flash-attention.
2. Clone [flash-attention](https://github.com/Dao-AILab/flash-attention) project or download its code to your machine. Change current directory to flash-attention: ````cd flash-attention``.
3. Install fused_mlp_lib. Refer to [here](https://github.com/Dao-AILab/flash-attention/tree/main/csrc/fused_dense_lib).
```python
cd csrc/fused_dense_lib && pip install .
```
4. Install layer_form. Refer to [here](https://github.com/Dao-AILab/flash-attention/tree/main/csrc/layer_norm).
```python
cd csrc/layer_norm && pip install .
```