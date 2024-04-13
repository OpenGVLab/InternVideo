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
TBD