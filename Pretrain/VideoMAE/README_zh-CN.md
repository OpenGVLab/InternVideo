# VideoMAE
代码继承自官方库 [VideoMAE](https://github.com/MCG-NJU/VideoMAE)，没有太多修改，主要增加多帧大分辨率部分，完善 data aug，修改以适应集群环境

## Installation
- python 3.6 or higher
- pytorch 1.8 or higher (推荐 pytorch 1.12 及以上，有效降低显存占用)
- timm==0.4.8/0.4.12
- deepspeed==0.5.8 (`DS_BUILD_OPS=1 pip install deepspeed`)
- TensorboardX
- decord
- einops
- opencv-python
- petrel sdk (用于读取 ceph 上数据，若直接读取本地磁盘不用安装)

pytorch 推荐 1.12 或以上的版本，能有效降低现存，timm 版本过高有 API 不兼容的风险，deepspeed 需要编译安装，由于服务器环境问题，部分算子无法安装，可以跳过（例如 `DS_BUILD_OPS=1 DS_BUILD_AIO=0 pip install deepspeed`）

## Data
data list 存放在 `/mnt/petrelfs/share_data/huangbingkun/data` 中， 可以将前缀 `s3://video_pub` 修改为可公共访问的 `/mnt/petrelfs/videointern`，直接从磁盘读取数据

## PreTrain
训练脚本在 `scripts/pretrain` 文件夹中，都为 slurm 训练版本，参数细节参考[VideoMAE-PRETRAIN](https://github.com/MCG-NJU/VideoMAE/blob/main/PRETRAIN.md)，运行示例：

```
bash scripts/pretrain/slurm_train_vit_h_hybrid_pt.sh ${JOB_NAME}
```

## Finetune
训练脚本在 `scripts/finetune` 文件夹中，都为 slurm 训练版本，参数细节参考[VideoMAE-FINETUNE]https://github.com/MCG-NJU/VideoMAE/blob/main/FINETUNE.md)，运行示例：

```
bash scripts/finetune/slurm_train_vit_h_k400_ft.sh ${JOB_NAME}
```

若只测试结果，在最后添加 `--eval` 即可
