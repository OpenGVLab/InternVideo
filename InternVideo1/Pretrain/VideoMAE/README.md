# VideoMAE
The code is modified from [VideoMAE](https://github.com/MCG-NJU/VideoMAE), and the following features have been added:

- support adjusting the input resolution and number of the frames when fine-tuning (The original offical codebase only support adjusting the number of frames)
- support applying repeated augmentation when pre-training

## Installation
- python 3.6 or higher
- pytorch 1.8 or higher
- timm==0.4.8/0.4.12
- deepspeed==0.5.8
- TensorboardX
- decord
- einops
- opencv-python
- (optional) petrel sdk (for reading the data on ceph)

## ModelZoo

| Backbone | Pretrain Data | Finetune Data | Epoch | \#Frame | Pre-train | Fine-tune | Top-1 | Top-5 |
| :------: | :-----: | :-----:| :---: | :-------: | :----------------------: | :--------------------: | :---: | :---: |
| ViT-B | UnlabeledHybrid | Kinetics-400 | 800 | 16 x 5 x 3 | [vit_b_hybrid_pt_800e.pth](https://pjlab-gvm-data.oss-cn-shanghai.aliyuncs.com/internvideo/pretrain/videomae/vit_b_hybrid_pt_800e.pth) | [vit_b_hybrid_pt_800e_k400_ft.pth](https://pjlab-gvm-data.oss-cn-shanghai.aliyuncs.com/internvideo/pretrain/videomae/vit_b_hybrid_pt_800e_k400_ft.pth) | 81.52 | 94.88 |
| ViT-B | UnlabeledHybrid | K710* | 800 | 16 x 5 x 3 | same as above | [vit_b_hybrid_pt_800e_k710_ft.pth](https://pjlab-gvm-data.oss-cn-shanghai.aliyuncs.com/internvideo/pretrain/videomae/vit_b_hybrid_pt_800e_k710_ft.pth) | 79.33 | 94.03 |
| ViT-B | UnlabeledHybrid | Something-Something V2 | 800 | 16 x 2 x 3 | same as above | [vit_b_hybrid_pt_800e_ssv2_ft.pth](https://pjlab-gvm-data.oss-cn-shanghai.aliyuncs.com/internvideo/pretrain/videomae/vit_b_hybrid_pt_800e_ssv2_ft.pth) | 71.22 | 93.31 |

Note: K710 is the union of different versions of Kinetics datasets (K400, K600, K700) where their label semantics are aligned and the duplicate videos with the validation sets are removed. K710 contains 658k training videos and 67k validation videos.

## Others
Please refer to [VideoMAE](https://github.com/MCG-NJU/VideoMAE) for Data, Pretrain and Finetune sections.
