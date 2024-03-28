import datetime
import logging
import time
import os
from os.path import join

import pandas as pd
import torch
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import wandb
from torch.utils.data import ConcatDataset

from dataset import MetaLoader, create_dataset, create_loader, create_sampler
from dataset.serialize import local_broadcast_process_authkey

from models.vindlu import VindLU
from models.vindlu_debug import VindLU_debug
from models.vindlu_vit import VindLU_VIT
from models.vindlu_vit_all import VindLU_VIT_ALL
from models.vindlu_vit_os import VindLU_VIT_OS
from models.vindlu_vit_mask import VindLU_VIT_MASK

from models.vindlu_blip_qformer import VindLU_BLIP_QFormer
from models.vindlu_blip_T5 import VindLU_BLIP_T5
from models.vindlu_blip_llama import VindLU_BLIP_Llama
from models.vindlu_videoclip import VindLU_VideoCLIP

from tasks.retrieval_utils import evaluation_wrapper as ret_eval_wrapper
from tasks.vqa_utils import evaluation_wrapper as qa_eval_wrapper
from tasks.shared_utils import get_media_types, setup_model
from utils.basic_utils import (MetricLogger, SmoothedValue,
                               remove_files_if_exist, setup_seed)
from utils.config_utils import setup_main
from utils.distributed import get_rank, get_world_size, is_main_process
from utils.logger import log_dict_to_wandb, setup_wandb

logger = logging.getLogger(__name__)


def train(
    model,
    train_loaders,
    optimizer,
    tokenizer,
    epoch,
    global_step,
    device,
    scheduler,
    scaler,
    config,
):
    model.train()

    metric_logger = MetricLogger(delimiter="  ")
    metric_logger.add_meter("sims", SmoothedValue(window=100, fmt="{value:.4f}"))

    requires_raw_text = config.criterion.get('mac_all', False) or \
        config.model.get("requires_raw_text", False)

    media_types = get_media_types(train_loaders)

    header = f"Train Epoch: [{epoch}]"
    log_freq = config.log_freq

    if config.distributed:
        for d in train_loaders:
            d.sampler.set_epoch(epoch)
    train_loader = MetaLoader(name2loader=dict(list(zip(media_types, train_loaders))))

    model.eval()
    iterator = metric_logger.log_every(train_loader, log_freq, header)
    for i, (media_type, (image, text, idx)) in enumerate(iterator):
        image = image.to(device, non_blocking=True)
        idx = idx.to(device, non_blocking=True)
        text_input = tokenizer(
            text,
            padding="max_length",
            truncation=True,
            max_length=config.inputs.max_txt_l[media_type],
            return_tensors="pt",
        ).to(
            device
        )  # change from "longest" to "max_length"

        with torch.cuda.amp.autocast(enabled=config.fp16, dtype=torch.bfloat16):
            if requires_raw_text:
                sims= model(image, text_input, idx=idx, raw_text=text, log_generation=(i % log_freq == 0), return_sims=True)
            else:
                sims= model(image, text_input, idx=idx, return_sims=True)

        metric_logger.update(temperature=sims.mean().item())

        if is_main_process() and config.wandb.enable and global_step % log_freq == 0:
            logs = metric_logger.get_global_avg_dict()
            log_dict_to_wandb(logs, step=global_step, prefix="train/")

        global_step += 1

        sims = sims.detach().diag().float().cpu().numpy()
        idx = idx.detach().cpu().numpy()
        # assert False, (sims.shape, idx.shape)

        rank = get_rank()
        os.makedirs("./sims", exist_ok=True)
        with open("./sims/sims_{}.txt".format(rank), "a+") as f:
            for i, s in zip(idx, sims):
                f.write("{}\t{:.4f}\n".format(i, s))

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    logger.info(f"Averaged stats: {metric_logger.global_avg()}")
    return global_step


def setup_dataloaders(config, mode="pt"):
    # train datasets, create a list of data loaders
    logger.info(f"Creating dataset for {mode}")
    train_datasets = create_dataset(f"{mode}_train", config)
    media_types = get_media_types(train_datasets)

    if config.distributed:
        num_tasks = get_world_size()
        global_rank = get_rank()
        samplers = create_sampler(
            train_datasets, [True] * len(media_types), num_tasks, global_rank
        )
    else:
        samplers = [None] * len(media_types)

    train_loaders = create_loader(
        train_datasets,
        samplers,
        batch_size=[config.inputs.batch_size[k] for k in media_types],
        num_workers=[config.num_workers] * len(media_types),
        is_trains=[True] * len(media_types),
        collate_fns=[None] * len(media_types),
    )  # [0]

    return train_loaders, media_types


def main(config):
    if is_main_process() and config.wandb.enable:
        run = setup_wandb(config)

    is_pretrain = config.mode == "pt"

    logger.info(f"train_file: {config.train_file}")

    setup_seed(config.seed + get_rank())
    device = torch.device(config.device)

    train_loaders, train_media_types = setup_dataloaders(
        config, mode=config.mode
    )
    num_steps_per_epoch = sum(len(d) for d in train_loaders)

    if config.scheduler.epochs < 1:
        logger.info(f"Num_epochs is set to {config.scheduler.epochs}, scale warmup_epochs accordingly, and set num_epochs to 1")
        config.scheduler.warmup_epochs = config.scheduler.warmup_epochs / config.scheduler.epochs
        config.scheduler.epochs = 1

    config.scheduler.num_training_steps = num_steps_per_epoch * config.scheduler.epochs
    config.scheduler.num_warmup_steps = num_steps_per_epoch * config.scheduler.warmup_epochs
    # set cudnn.benchmark=True only when input size is fixed
    # https://discuss.pytorch.org/t/what-does-torch-backends-cudnn-benchmark-do/5936/3
    cudnn.benchmark = len(train_media_types) == 1

    model_cls = eval(config.model.get('model_cls', 'VindLU'))
    find_unused_parameters = True
    if any([x in config.model.get('model_cls', 'VindLU') for x in ['VindLU_BLIP', 'VindLU_VideoCLIP']]):
        find_unused_parameters = False
    (
        model,
        model_without_ddp,
        optimizer,
        scheduler,
        scaler,
        tokenizer,
        start_epoch,
        global_step,
    ) = setup_model(
        config,
        model_cls=model_cls,
        has_decoder=False,
        pretrain=is_pretrain,
        find_unused_parameters=find_unused_parameters,
        num_steps_per_epoch=num_steps_per_epoch,
    )
    if is_main_process() and config.wandb.enable:
        wandb.watch(model)

    best = 0
    best_epoch = 0

    logger.info("Start training")
    start_time = time.time()
    epoch = 0
    global_step = train(
        model,
        train_loaders,
        optimizer,
        tokenizer,
        epoch,
        global_step,
        device,
        scheduler,
        scaler,
        config,
    )


if __name__ == "__main__":
    cfg = setup_main()
    local_broadcast_process_authkey()
    main(cfg)
