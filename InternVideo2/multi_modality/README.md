# Multi-modality of InternVideo2

## Installation

Please follow the installation instructions in [INSTALL](./INSTALL.md).

>The codebase support using [wandb](https://wandb.ai/) to monitor training. If you want to use wandb, you will need to set up it following [this very short instruction](https://docs.wandb.ai/quickstart#1.-set-up-wandb), and also set `wandb.enable` in the config to be `True`. `wandb.entity` and `wandb.project` should also be set.

## Datasets

You can find the dataset instructions in [DATASET](DATASET.md).

## Model ZOO

You can find all the models and the scripts in [MODEL_ZOO](./MODEL_ZOO.md).

## Demo of Using InternVideo2 in Your Work
We give a short instructions of accessing and utilizing InternVideo2-stage2 in [demo.ipynb](./demo/demo.ipynb).

## Pre-Training

We use [InternVL](https://github.com/OpenGVLab/InternVL/) pretrained model as the teacher by default

For training, you can simply run the pretraining scripts in `scripts/pretraining` as follows:
```shell
bash scripts/pretraining/stage2/1B/run.sh
```

:warning: **Notes:**
1. Set `data_dir` and `your_data_path` like `your_webvid_path` in [data.py](./configs/data.py) before running the scripts.
2. Set `vision_encoder.pretrained` in `vision_encoder.pretrained` in the corresponding config files.
3. Set `--rdzv_endpoint` to your `MASTER_NODE:MASTER_PORT`. You can also use the following commond to automatically set it:
    ```shell
    MASTER_NODE=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
    ALL_NODES=$(scontrol show hostnames "$SLURM_JOB_NODELIST")
    MASTER_PORT=$((10000 + $RANDOM % 100))
    torchrun --rdzv_endpoint=${MASTER_NODE}:10068 $@
    ```
4. `save_latest=True` will automatically save the latest checkpoint while training.
5. `auto_resume=True` will automatically loaded the best or latest checkpoint while training.

## CLIP Post-pretraining

We use the text encoder from [InternVL-CLIP](https://github.com/OpenGVLab/InternVL/tree/main/clip_benchmark).

For training, you can simply run the pretraining scripts in `scripts/pretraining` as follows:
```shell
bash scripts/pretraining/clip/1B/run.sh
```

:warning: **Notes:**
1. Download [chinese_alpaca_lora_7b](https://github.com/OpenGVLab/InternVL/tree/main/clip_benchmark/clip_benchmark/models/internvl_c_pytorch/chinese_alpaca_lora_7b) and set the `llama_path` and `tokenizer_path` in `config.py`.
2. Download [InternVideo2-stage2_1b-224p-f4.pt](https://huggingface.co/OpenGVLab/InternVideo2/blob/main/InternVideo2-stage2_1b-224p-f4.pt) and set `vision_ckpt_path` in `config.py`.
3. Download [internvl_c_13b_224px](https://huggingface.co/OpenGVLab/InternVL/blob/main/internvl_c_13b_224px.pth) and set `text_ckpt_path` in `config.py`.


## Zero-shot Evaluation

For zero-shot evaluation, you can simply run the pretraining scripts in `scripts/evaluation` as follows:
```shell
bash scripts/evaluation/stage2/zero_shot/1B/eval_msrvtt.sh
```
When evaluating, you can choose to turn off deepspeed and the performance will fluctuate slightly from the reported result (around 0.2), modify the value in the corresponding config file (e.g. scripts\evaluation\stage2\zero_shot\1B\config_msrvtt.py):
```shell
deepspeed = dict(
    enable=False,
    stage=1,
)
```

:warning: **Notes:**
1. Set `pretrained_path=your_model_path` in the running scripts before running the scripts.
2. Set `zero_shot=True` and `evaluate=True` for zero-shot evaluation 

## Finetuning

Coming soon.

