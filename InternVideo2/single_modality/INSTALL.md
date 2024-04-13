# Installation

## Requirements

We mainly follow [UMT](https://github.com/OpenGVLab/Unmasked_Teacher) to prepare the enviroment.

```shell
pip install -r requirements.txt
```

We follow UMT to set `--epochs 201` to avoid the potential interrupt in the last epoch.

> We observed accidental interrupt in the last epoch when conducted the pre-training experiments on V100 GPUs (PyTorch 1.6.0). This interrupt is caused by the scheduler of learning rate. We naively set --epochs 801 to walk away from issue.

## Note

To run InternVideo2 pretraining, you have to prepare the weights of the **[InternVL-6B visual encoder](https://huggingface.co/OpenGVLab/InternVL/blob/main/internvl_c_13b_224px.pth)** and **[VideoMAEv2-g](https://github.com/OpenGVLab/VideoMAEv2/blob/master/docs/MODEL_ZOO.md)**, and set the `your_model_path` in [internvl_clip_vision.py](./models/internvl_clip_vision.py) and [videomae.py](./models/videomae.py).