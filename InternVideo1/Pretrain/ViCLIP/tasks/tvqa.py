import copy
import datetime
import logging
import os
import time
from os.path import join

import pandas as pd
import torch
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import wandb
from omegaconf import OmegaConf

from models.vindlu_tvqa import VindLU_TVQA
from tasks.pretrain import setup_dataloaders
from tasks.shared_utils import setup_model
from utils.basic_utils import (MetricLogger, SmoothedValue, flat_list_of_lists,
                               save_json, setup_seed)
from utils.config_utils import setup_main
from utils.distributed import get_rank, is_main_process
from utils.logger import log_dict_to_wandb, setup_wandb

logger = logging.getLogger(__name__)


def train(
    model,
    train_loader,
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
    metric_logger.add_meter("lr", SmoothedValue(window=1, fmt="{value:.6f}"))
    loss_names = ["loss_qa"]
    for name in loss_names:
        metric_logger.add_meter(f"{name}", SmoothedValue(window=1, fmt="{value:.4f}"))

    header = f"Train Epoch: [{epoch}]"
    log_freq = config.log_freq

    if config.distributed:
        train_loader.sampler.set_epoch(epoch)

    iterator = metric_logger.log_every(train_loader, log_freq, header)
    for i, (image, text, answer_idx, qid) in enumerate(iterator):
        image = image.to(device, non_blocking=True)
        answer_idx = answer_idx.to(device, non_blocking=True)
        text = flat_list_of_lists(zip(*text))
        text_input = tokenizer(
            text,
            padding="max_length",
            truncation=True,
            max_length=config.max_txt_l,
            return_tensors="pt",
        ).to(device)

        with torch.cuda.amp.autocast(enabled=config.fp16, dtype=torch.bfloat16):
            loss_dict = model(image, text_input, answer_idx, train=True)
            loss = sum(loss_dict.values())

        optimizer.zero_grad()
        scaler.scale(loss).backward()
        if config.optimizer.max_grad_norm > 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.optimizer.max_grad_norm)
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()

        # logging
        for name in loss_names:
            value = loss_dict[name]
            value = value if isinstance(value, float) else value.item()
            metric_logger.update(**{f"{name}": value})
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])

        if is_main_process() and config.wandb.enable and global_step % log_freq == 0:
            logs = metric_logger.get_global_avg_dict()
            log_dict_to_wandb(logs, step=global_step, prefix="train/")

        global_step += 1

        if config.debug and (i + 1) % 5 == 0:
            break

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    logger.info(f"Averaged train stats: {metric_logger.global_avg()}")
    return global_step


@torch.no_grad()
def evaluation(model, data_loader, tokenizer, device, config):
    model.eval()

    metric_logger = MetricLogger(delimiter="  ")
    header = "[evaluation] Generating answers:"
    log_freq = config.log_freq // 2

    gt_answers = []
    pred_answers = []
    iterator = metric_logger.log_every(data_loader, log_freq, header)
    for i, (image, text, answer_idx, qid) in enumerate(iterator):
        image = image.to(device, non_blocking=True)
        text = flat_list_of_lists(zip(*text))
        text_input = tokenizer(
            text,
            padding="max_length",
            truncation=True,
            max_length=config.max_txt_l,
            return_tensors="pt",
        ).to(device)

        with torch.cuda.amp.autocast(enabled=config.fp16, dtype=torch.bfloat16):
            _preds = model(image, text_input, answer_idx, train=False)

        pred_answers.append(_preds)
        gt_answers.append(answer_idx)

    pred_answers = torch.cat(pred_answers, 0)  # (N, )
    gt_answers = torch.cat(gt_answers, 0)  # (N,)
    acc = torch.mean((pred_answers == gt_answers).to(float))
    return float(acc)


def main(config):
    if is_main_process() and config.wandb.enable:
        run = setup_wandb(config)

    logger.info(f"train_file: {config.train_file}")

    setup_seed(config.seed + get_rank())
    device = torch.device(config.device)
    cudnn.benchmark = True

    train_loaders, test_name2loaders, train_media_types = setup_dataloaders(
        config, mode="tvqa"
    )
    train_loader = train_loaders[0]
    num_steps_per_epoch = len(train_loader)
    config.scheduler.num_training_steps = num_steps_per_epoch * config.scheduler.epochs
    config.scheduler.num_warmup_steps = num_steps_per_epoch * config.scheduler.warmup_epochs

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
        model_cls=VindLU_TVQA,
        has_decoder=False,
        pretrain=False,
        find_unused_parameters=True,
    )
    if is_main_process() and config.wandb.enable:
        wandb.watch(model)

    best = 0
    best_epoch = 0

    logger.info("Start " + "evaluation" if config.evaluate else "training")
    start_time = time.time()
    for epoch in range(start_epoch, config.scheduler.epochs):
        if not config.evaluate:
            global_step = train(
                model,
                train_loader,
                optimizer,
                tokenizer,
                epoch,
                global_step,
                device,
                scheduler,
                scaler,
                config,
            )

        with torch.cuda.amp.autocast(enabled=config.fp16, dtype=torch.bfloat16):
            eval_res = {}
            for test_name, test_loader in test_name2loaders.items():
                if test_name not in config.test_types:
                    logger.info(
                        f"Skip eval {test_name} split. All test_types {config.test_types}"
                    )
                    continue
                res = evaluation(model_without_ddp, test_loader, tokenizer, device, config)
                eval_res[test_name] = round(res * 100, 2)

        if is_main_process():
            if config.wandb.enable:
                log_dict_to_wandb(eval_res, step=global_step, prefix="")

            if config.stop_key is not None and config.stop_key in eval_res:
                cur_acc = eval_res[config.stop_key]
            else:  # None
                cur_acc = best + 1  # save the last as the best
            logger.info(f"Epoch {epoch}")
            logger.info(f"{eval_res}")
            save_json(eval_res, join(config.output_dir, "eval_res_latest.json"))

            if not config.evaluate and cur_acc > best:
                save_obj = {
                    "model": model_without_ddp.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict(),
                    "scaler": scaler.state_dict(),
                    "config": config,
                    "epoch": epoch,
                    "global_step": global_step,
                }
                eval_file = "eval_res_best.json"
                save_json(eval_res, join(config.output_dir, eval_file))
                torch.save(save_obj, join(config.output_dir, "ckpt_best.pth"))
                best = cur_acc
                best_epoch = epoch
            if config.evaluate:
                eval_file = "eval_res.json"
                save_json(eval_res, join(config.output_dir, eval_file))

        if config.evaluate or config.debug:
            break

        dist.barrier()

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    logger.info(f"Training time {total_time_str}")
    logger.info(f"best epoch {best_epoch} [config.stop_key {config.stop_key}]")
    logger.info(f"Checkpoints and Logs saved at {config.output_dir}")

    if is_main_process() and config.wandb.enable:
        run.finish()


def eval_after_training(train_config):
    # general config for all
    train_config.wandb.enable = False
    train_config.evaluate = True
    train_config.pretrained_path = join(train_config.output_dir, "ckpt_best.pth")

    eval_config = copy.deepcopy(train_config)
    eval_config.test_types = list(eval_config.test_file.keys())
    eval_config.output_dir = join(eval_config.output_dir, f"eval_after_training")
    eval_config.result_dir = eval_config.output_dir
    if is_main_process():
        os.makedirs(eval_config.output_dir, exist_ok=False)
        OmegaConf.save(eval_config, open(join(eval_config.output_dir, "config.yaml"), "w"))
    logger.info(f"===========> START eval_after_training [{eval_config.test_types}]")
    main(eval_config)


if __name__ == "__main__":
    cfg = setup_main()
    main(cfg)
    if not cfg.evaluate:
        eval_after_training(cfg)
