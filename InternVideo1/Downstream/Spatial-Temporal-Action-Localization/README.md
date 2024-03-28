# VideoMAE Installation

## Environment Configuration

The codebase is mainly built with following libraries:

- Python 3.6 or higher
- [PyTorch](https://pytorch.org/) and [torchvision](https://github.com/pytorch/vision). <br>
  We can successfully reproduce the main results in two settings:<br>
  Tesla **A100** (40G): CUDA 11.1 + PyTorch 1.8.0 + torchvision 0.9.0
  Tesla **V100** (32G): CUDA 10.1 + PyTorch 1.6.0 + torchvision 0.7.0
<br> The torch version here has a great impact on the results. It is recommended to configure the environment according to such settings or a newer version.</br>
- [timm==0.4.8/0.4.12](https://github.com/rwightman/pytorch-image-models)
- [deepspeed==0.5.8](https://github.com/microsoft/DeepSpeed)
- [TensorboardX](https://github.com/lanpa/tensorboardX)
- [decord](https://github.com/dmlc/decord)
- [einops](https://github.com/arogozhnikov/einops)
- [av](https://github.com/PyAV-Org/PyAV)
- [tqdm](https://github.com/tqdm/tqdm)

<br>We recommend to setup the environment with Anaconda, the step-by-step installation script is shown below.</br>
```shell
conda create -n VideoMAE_ava python=3.7
conda activate VideoMAE_ava

#install pytorch with the same cuda version as in your environment
pip install torch==1.11.0+cu113 torchvision==0.12.0+cu113 torchaudio==0.11.0 --extra-index-url https://download.pytorch.org/whl/cu113

conda install av -c conda-forge
conda install cython
```


## Data Preparation

### AVA2.2
The code combines [VideoMAE](https://github.com/MCG-NJU/VideoMAE) and [Alphaction](https://github.com/MVIG-SJTU/AlphAction), and the preparation of AVA data refers to the [data preparation](https://github.com/MVIG-SJTU/AlphAction/blob/master/DATA.md) of Alphaction. If you only need to train and test on the AVA dataset, you do not need to prepare the Kinetics dataset.

### Kinetics

### Other_files
`video_map.npy` : Mapping of video id and corresponding video path

`ak_val_gt.csv` : The ground truth of the val-set of ava-kinetics


### AVA-Kinetics file
In order to facilitate everyone to download together, we have organized the annotation files we used, which are available for download in [OneDrive](https://1drv.ms/u/s!AjZXNAXrK-eti2zl_JNeQbmizFbC?e=VjdMJ3). It should be noted that the files we use may be slightly different from the officially provided files, especially for kinetics, the annotations we use The version may be older, and some videos may be different from what you are downloading now.

## Train
Here is a script that uses the ava-kinetics dataset for training and eval on the ava dataset
```bash
MODEL_PATH='YOUR_PATH/PRETRAIN_MODEL.pth'
OUTPUT_DIR='YOUR_PATH/OUTPUT_DIR'
python -m torch.distributed.launch --nproc_per_node=8 \
      --master_port 12320 --nnodes=8 \
      --node_rank=0 --master_addr=$ip_node_0 \
      run_class_finetuning.py \
      --model vit_large_patch16_224 \
      --finetune ${MODEL_PATH} \
      --log_dir ${OUTPUT_DIR} \
      --output_dir ${OUTPUT_DIR} \
      --batch_size 8 \
      --update_freq 1 \
      --num_sample 1 \
      --input_size 224 \
      --save_ckpt_freq 1 \
      --num_frames 16 \
      --sampling_rate 4 \
      --opt adamw \
      --lr 0.00025 \
      --opt_betas 0.9 0.999 \
      --weight_decay 0.05 \
      --epochs 30 \
      --data_set "ava-kinetics" \
      --enable_deepspeed \
      --val_freq 30 \
      --drop_path 0.2\
```


* SLURM ENV
```bash
MODEL_PATH='YOUR_PATH/PRETRAIN_MODEL.pth'
OUTPUT_DIR='YOUR_PATH/OUTPUT_DIR'
PARTITION=${PARTITION:-"video"}
GPUS=${GPUS:-32}
GPUS_PER_NODE=${GPUS_PER_NODE:-8}
CPUS_PER_TASK=${CPUS_PER_TASK:-12}
SRUN_ARGS=${SRUN_ARGS:-""}
PY_ARGS=${@:2}
srun -p video \
     --gres=gpu:${GPUS_PER_NODE} \
     --ntasks=${GPUS} \
     --ntasks-per-node=${GPUS_PER_NODE} \
     --cpus-per-task=${CPUS_PER_TASK} \
     ${SRUN_ARGS} \
     python -u run_class_finetuning.py \
      --model vit_large_patch16_224 \
      --finetune ${MODEL_PATH} \
      --log_dir ${OUTPUT_DIR} \
      --output_dir ${OUTPUT_DIR} \
      --batch_size 8 \
      --update_freq 1 \
      --num_sample 1 \
      --input_size 224 \
      --save_ckpt_freq 1 \
      --num_frames 16 \
      --sampling_rate 4 \
      --opt adamw \
      --lr 0.00025 \
      --opt_betas 0.9 0.999 \
      --weight_decay 0.05 \
      --epochs 30 \
      --data_set "ava" \
      --enable_deepspeed \
      --val_freq 30 \
      --drop_path 0.2\
      ${PY_ARGS}
```

## Eval
* SLURM ENV

```shell
DATA_PATH='YOUR_PATH/list_kinetics-400'   #it can be any string in our task
MODEL_PATH='YOUR_PATH/PRETRAIN_MODEL.pth'
OUTPUT_DIR='YOUR_PATH/OUTPUT_DIR'
PARTITION=${PARTITION:-"video"}
GPUS=${GPUS:-32}
GPUS_PER_NODE=${GPUS_PER_NODE:-8}
CPUS_PER_TASK=${CPUS_PER_TASK:-12}
SRUN_ARGS=${SRUN_ARGS:-""}
PY_ARGS=${@:2}
srun -p video \
     --gres=gpu:${GPUS_PER_NODE} \
     --ntasks=${GPUS} \
     --ntasks-per-node=${GPUS_PER_NODE} \
     --cpus-per-task=${CPUS_PER_TASK} \
     ${SRUN_ARGS} \
     python -u run_class_finetuning.py \
      --model vit_large_patch16_224 \
      --data_path ${DATA_PATH} \
      --finetune ${MODEL_PATH} \
      --log_dir ${OUTPUT_DIR} \
      --output_dir ${OUTPUT_DIR} \
      --batch_size 4 \
      --update_freq 1 \
      --num_sample 1 \
      --input_size 224 \
      --save_ckpt_freq 1 \
      --num_frames 16 \
      --sampling_rate 4 \
      --opt adamw \
      --lr 0.00025 \
      --opt_betas 0.9 0.999 \
      --weight_decay 0.05 \
      --epochs 30 \
      --data_set "ava" \
      --enable_deepspeed \
      --val_freq 30 \
      --drop_path 0.2\
      --eval \
      ${PY_ARGS}
```