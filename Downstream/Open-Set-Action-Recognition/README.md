# Open Set Action Recognition

## Table of Contents
1. [Introduction](#introduction)
1. [Installation](#installation)
1. [Datasets](#datasets)
1. [Testing](#testing)
1. [Training](#training)
1. [Model Zoo](#model-zoo)


## Introduction
VideoIntern not only recognizes known classes accurately but also has a strong perception ability for unknown classes that are out of training classes. This repo is one of the generalization tasks——Open Set Action Recognition. Specifically, we finetune VideoMAE backbone with a linear classification head on UCF-101 dataset from the evidential deep learning (EDL) perspective, without any model calibration methods used in [DEAR](https://github.com/Cogito2012/DEAR). Our VideoIntern model achieves significant and consistent performance gains compared to multiple action recognition backbones (i.e., I3D, TSM, SlowFast, TPN), which are trained in the DEAR way, with HMDB-51 or MiT-v2 dataset as the unknown.    

## Installation
This repo is developed from [MMAction2](https://github.com/open-mmlab/mmaction2) codebase.

### Installation Steps
a. Create a conda virtual environment of this repo, and activate it:

```shell
conda create -n OSAR python=3.7 -y
conda activate OSAR
```

b. Install PyTorch and TorchVision following the [official instructions](https://pytorch.org/), e.g.,

```shell
conda install pytorch=1.7.0 cudatoolkit=11.0 torchvision=0.8.0 -c pytorch
```
c. Install mmcv, we recommend you to install the pre-build mmcv as below.

```shell
pip install mmcv-full==1.2.2 -f https://download.openmmlab.com/mmcv/dist/cu110/torch1.7.0/index.html
```
**Important:** If you have already installed `mmcv` and try to install `mmcv-full`, you have to uninstall `mmcv` first by running `pip uninstall mmcv`. Otherwise, there will be `ModuleNotFoundError`.

d. Clone the source code of this repo:

```shell
git clone https://github.com/VideoIntern/Open-Set-Action-Recognition.git Open-Set-Action-Recognition-main
cd Open-Set-Action-Recognition-main
```
e. Install build requirements and then install OSAR.

```shell
pip install -r requirements/build.txt
python setup.py develop
```

If no error appears in your installation steps, then you are all set!

## Datasets

This repo uses standard video action datasets, i.e., UCF-101 for closed set training, and HMDB-51 and MiT-v2 test sets as two different unknowns. Please refer to the default [MMAction2 dataset setup steps](/tools/data/ucf101/README.md) to setup these three datasets correctly.

**Note**: You can just ignore the `Step 3. Extract RGB and Flow` in the referred setup steps since all codes related to our paper do not rely on extracted frames and optical flow. This will save you large amount of disk space!

## Testing

To test our pre-trained models (see the [Model Zoo](#model-zoo)), you need to download a model file and unzip it under `work_dir`. Let's take the `I3D`-based DEAR model as an example. First, download the [pre-trained I3D-based models](https://drive.google.com/drive/folders/1TguABfmy0PE6jx9fflePQySe3jXWnsc0?usp=sharing), where the full DEAR model is saved in the folder `finetune_ucf101_i3d_edlnokl_avuc_debias`. The following directory tree is for your reference to place the downloaded files.
```shell
work_dirs    
├── mae
│    ├── finetune_ucf101_mae_edlnokl
│    │   └── latest.pth

```
a. Get Uncertainty Threshold.
The threshold value of one model will be reported.
```shell
cd experiments/mae
# run the thresholding with BATCH_SIZE=16 on 8 GPUs
bash run_get_mae_threshold.sh edlnokl 16 8
```

b. Out-of-Distribution Detection.
The uncertainty distribution figure of a specified model will be reported.
```shell
cd experiments/mae
bash run_ood_mae_dist_detection.sh HMDB edlnokl 8
```

c. Compute AUROC.
The AUROC score of a specified model will be reported.
```shell
cd experiments/mae/results
python compute_auroc.py
```

## Training
 
```shell
cd experiments/mae
bash finetune_mae_edlnokl_ucf101.sh 8
```

## Model Zoo

The pre-trained weights (checkpoints) are available below.
| Model | Checkpoint | Train Config | Test Config |  Open Set AUC (%) | Closed Set ACC (%) |
|:--|:--:|:--:|:--:|:--:|:--:|
|I3D + DEAR |[ckpt](https://drive.google.com/file/d/1oRNBH0aAhFpcJSBqWlT4x0ru7iHfdndW/view?usp=sharing)| [train](configs/recognition/i3d/finetune_ucf101_i3d_edlnokl_avuc_debias.py) | [test](configs/recognition/i3d/inference_i3d_enn.py) | 77.08 / 81.54 | 93.89 |
|TSM + DEAR | [ckpt](https://drive.google.com/file/d/1TM1c28jRyZpOrWqwaQPYXFBZJXHQp__9/view?usp=sharing)| [train](configs/recognition/tsm/finetune_ucf101_tsm_edlnokl_avuc_debias.py) | [test](configs/recognition/tsm/inference_tsm_enn.py) | 78.65 / 83.92 | 94.48 |
|TPN + DEAR | [ckpt](https://drive.google.com/file/d/1jorfFMMzWd5xDCfZsemoWD8Rg7DbH16u/view?usp=sharing)| [train](configs/recognition/tpn/tpn_slowonly_edlloss_nokl_avuc_debias_r50_8x8x1_150e_kinetics_rgb.py) | [test](configs/recognition/tpn/inference_tpn_slowonly_enn.py) | 79.23 / 81.80 | 96.30 |
|SlowFast + DEAR |[ckpt](https://drive.google.com/file/d/13LNRv0BYkVfzCA95RB5dCp53MmErRL5D/view?usp=sharing)| [train](configs/recognition/slowfast/finetune_ucf101_slowfast_edlnokl_avuc_debias.py) | [test](configs/recognition/slowfast/inference_slowfast_enn.py)  | 82.94 / 86.99 | 96.48 |
|InternVideo-B + EDL |[ckpt](https://drive.google.com/file/d/1lW1mHCbyfi0tvIxAjVzjr3g-AgK61ND3/view?usp=share_link)| [train](configs/recognition/mae/finetune_ucf101_mae_edlnokl.py) | [test](configs/recognition/mae/inference_mae_enn.py)  | 83.21 / 88.98 | 96.91 |
|InternVideo-L + EDL |[ckpt](https://drive.google.com/file/d/1lW1mHCbyfi0tvIxAjVzjr3g-AgK61ND3/view?usp=share_link)| [train](configs/recognition/mae/finetune_ucf101_mae_edlnokl.py) | [test](configs/recognition/mae/inference_mae_enn.py)  | 83.82 / 91.13 | 97.36 |
|InternVideo-H + EDL |[ckpt](https://drive.google.com/file/d/1lW1mHCbyfi0tvIxAjVzjr3g-AgK61ND3/view?usp=share_link)| [train](configs/recognition/mae/finetune_ucf101_mae_edlnokl.py) | [test](configs/recognition/mae/inference_mae_enn.py)  | 85.48 / 91.85 | 97.89 |

For the pretrained MAE model, please download it in the [Google Drive](https://drive.google.com/file/d/1iVb7c3onYPjIv5ResMRbIoxCVlsp5YCr/view?usp=share_link).


## License

See [Apache-2.0 License](/LICENSE)

## Acknowledgement

In addition to the MMAction2 codebase, this repo contains modified codes from:
 - [pytorch-classification-uncertainty](https://github.com/dougbrion/pytorch-classification-uncertainty): for implementation of the [EDL (NeurIPS-2018)](https://arxiv.org/abs/1806.01768).
 - [ARPL](https://github.com/iCGY96/ARPL): for implementation of baseline method [RPL (ECCV-2020)](https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123480511.pdf).
 - [OSDN](https://github.com/abhijitbendale/OSDN): for implementation of baseline method [OpenMax (CVPR-2016)](https://vast.uccs.edu/~abendale/papers/0348.pdf).
 - [bayes-by-backprop](https://github.com/nitarshan/bayes-by-backprop/blob/master/Weight%20Uncertainty%20in%20Neural%20Networks.ipynb): for implementation of the baseline method Bayesian Neural Networks (BNNs).
 - [rebias](https://github.com/clovaai/rebias): for implementation of HSIC regularizer used in [ReBias (ICML-2020)](https://arxiv.org/abs/1910.02806)

We sincerely thank the owners of all these great repos!
