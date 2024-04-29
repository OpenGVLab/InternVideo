import copy
import datetime
import logging
import os
import time
import json
from os.path import join
from torchnet import meter

import pandas as pd
import torch
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import wandb
import torch.nn.functional as F
from einops import rearrange

from dataset import MetaLoader, create_dataset, create_loader, create_sampler
from models.utils import tile
from models import *
from tasks_clip.shared_utils import get_media_types, setup_model
from utils.basic_utils import MetricLogger, SmoothedValue, setup_seed, flat_list_of_lists, save_json
from utils.config import Config
from utils.config_utils import setup_main
from utils.distributed import get_rank, get_world_size, is_main_process
from utils.logger import log_dict_to_wandb, setup_wandb

logger = logging.getLogger(__name__)


def get_sim_for_each_question(model, pooled_image_feat, pooled_text_feat, model_cls):
    """TODO: Docstring for get_sim_for_each_question.

    Args:
        model (TODO): TODO
        pooled_image_feat (torch.Tensor): Shape: [b, c]
        pooled_text_feat (torch.Tensor): Shape: [b, n, c]. n is the number of answer candidates.

    Returns: TODO

    """
    image_feat = F.normalize(pooled_image_feat, dim=-1).to(torch.float32)
    text_feat = F.normalize(pooled_text_feat, dim=-1).to(torch.float32)
    sim = torch.matmul(image_feat.unsqueeze(1), rearrange(text_feat, "b n c -> b c n"))  # [b, 1, n]
    sim = sim[:, 0] / model.temp  # [b, n]
    sim = F.softmax(sim, dim=1)  # [b, n]
    return sim


def main_with_ensemble(config, test_loader, model_without_ddp, tokenizer, data_type):
    logger.info(f"test_file: {config.test_file}")

    setup_seed(config.seed + get_rank())
    device = torch.device(config.device)
    cudnn.benchmark = True

    config.scheduler.num_training_steps = 10
    config.scheduler.num_warmup_steps = 10
    model = model_without_ddp
    model.eval()

    logger.info("Start " + "evaluation" if config.evaluate else "training")
    metric_logger = MetricLogger(delimiter="  ")
    iterator = metric_logger.log_every(test_loader, 5, "Evaluation: ")
    num_options_per_q = 157
    all_gt_answers = None
    all_preds = None
    predictions = []
    with torch.cuda.amp.autocast(enabled=config.use_half_precision, dtype=data_type), torch.no_grad():
        for image, text, ans, ann in iterator:
            image = image.to(device, non_blocking=True)  # bsz
            if all_gt_answers is None:
                all_gt_answers = ans
            else:
                all_gt_answers = torch.cat([all_gt_answers, ans], dim=0) # bsz*?
            text = flat_list_of_lists(list(zip(*text)))  # List(str), len=bsz*157
            text_input = tokenizer(text).to(device)  # bsz*157

            # encode text
            pooled_text_feat = model.encode_text(text_input) # [b*157, c]
            # encode image
            pooled_image_feat = model.encode_vision(image, test=True) # [bsz, c]

            # contrastive score
            pooled_text_feat = rearrange(pooled_text_feat, "(b n) c -> b n c", n=num_options_per_q)
            score = get_sim_for_each_question(model, pooled_image_feat, pooled_text_feat, model_cls=config.model.model_cls)  # [b, n]

            # assemble predictions
            if all_preds is None:
                all_preds = score
            else:
                all_preds = torch.cat([all_preds, score], dim=0)
            for q_idx in range(len(score)):  # bsz
                _pred = dict(
                    video=ann["video"][q_idx],
                    answer=ann["answer"][q_idx],
                    pred_scores=score[q_idx].cpu().numpy(),  # (bsz, 157)
                )
                predictions.append(_pred)

    # generate ft mactrix
    gt = all_gt_answers.long()
    map_meter = meter.mAPMeter()
    map_meter.add(all_preds, gt)
    ap = map_meter.value()
    ap = float(ap) * 100

    eval_res = {"map": round(ap, 2)}
    logger.info(f"\n{eval_res}")
    save_json(eval_res, join(config.output_dir, "eval_res.json"))
    torch.save(predictions, join(config.output_dir, "prediction_scores.pth"))
    return eval_res


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
    data_type
):
    model.train()

    metric_logger = MetricLogger(delimiter="  ")
    metric_logger.add_meter("lr", SmoothedValue(window=1, fmt="{value:.6f}"))
    metric_logger.add_meter("temperature", SmoothedValue(window=1, fmt="{value:.4f}"))
    loss_names = ["loss_" + k for k, v in config.criterion.loss_weight.items() if v != 0]

    media_types = [loader.dataset.media_type for loader in train_loaders]
    for name in loss_names:
        for m in media_types:
            metric_logger.add_meter(f"{m}-{name}", SmoothedValue(window=1, fmt="{value:.4f}"))

    header = f"Train Epoch: [{epoch}]"
    log_freq = config.log_freq

    if config.distributed:
        for d in train_loaders:
            d.sampler.set_epoch(epoch)
    train_loader = MetaLoader(name2loader=dict(list(zip(media_types, train_loaders))))

    model_without_ddp = model.module if config.distributed else model
    iterator = metric_logger.log_every(train_loader, log_freq, header)
    for i, (media_type, (image, text, idx)) in enumerate(iterator):
        image = image.to(device, non_blocking=True)
        idx = idx.to(device, non_blocking=True)
        text_input = tokenizer(
            text,
            padding="max_length",
            truncation=True,
            max_length=config.max_txt_l,
            return_tensors="pt",
        ).to(device)

        with torch.cuda.amp.autocast(enabled=config.use_half_precision, dtype=data_type):
            loss_dict = model(image, text_input, idx=idx)
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
            metric_logger.update(**{f"{media_type}-{name}": value})
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])
        metric_logger.update(temperature=model_without_ddp.temp.item())

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


def main(config):
    if is_main_process() and config.wandb.enable:
        run = setup_wandb(config)

    logger.info(f"config: \n{config}")
    logger.info(f"train_file: {config.train_file}")

    setup_seed(config.seed + get_rank())
    device = torch.device(config.device)
    cudnn.benchmark = True

    # create train loader
    train_datasets = create_dataset("ret_train", config)
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
    )

    num_steps_per_epoch = sum(len(d) for d in train_loaders)
    config.scheduler.num_training_steps = num_steps_per_epoch * config.scheduler.epochs
    config.scheduler.num_warmup_steps = num_steps_per_epoch * config.scheduler.warmup_epochs

    model_cls = eval(config.model.get('model_cls', 'InternVideo2_CLIP'))
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
        pretrain=False,
        # find_unused_parameters=True,
        find_unused_parameters=False,
    )
    if is_main_process() and config.wandb.enable:
        wandb.watch(model)
    
    # create test dataloader
    test_dataset = create_dataset("mc_new_test", config)
    test_loader = create_loader(
        [test_dataset],
        [None],
        batch_size=[config.inputs.batch_size_test.video],
        num_workers=[config.num_workers],
        is_trains=[False],
        collate_fns=[None],
    )[0]

    best = 0
    best_epoch = 0

    if config.get('use_bf16', True):
        data_type = torch.bfloat16
    else:
        data_type = torch.float16

    logger.info("Start " + "evaluation" if config.evaluate else "training")
    start_time = time.time()
    for epoch in range(start_epoch, config.scheduler.epochs):
        if not config.evaluate:
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
                data_type=data_type
            )

        # save checkpoint befor evaluation
        # only save those with gradient
        if not config.evaluate:
            if hasattr(config, "deepspeed") and config.deepspeed.enable:
                if config.get("save_latest", False):
                    tag = "ckpt_latest.pth"
                else:
                    tag = f"ckpt_{epoch:02d}.pth"
                model.save_checkpoint(config.output_dir, tag=tag, save_latest=False, exclude_frozen_parameters=True)

            elif is_main_process():
                state_dict = model_without_ddp.state_dict()
                param_grad_dict = {
                    k: v.requires_grad for (k, v) in model_without_ddp.named_parameters()
                }
                for k in list(state_dict.keys()):
                    if k in param_grad_dict.keys() and not param_grad_dict[k]:
                        # delete parameters that do not require gradient
                        logger.info(f"Not saving {k}")
                        del state_dict[k]

                save_obj = {
                    "model": state_dict,
                    "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict(),
                    "scaler": scaler.state_dict(),
                    "config": config,
                    "epoch": epoch,
                    "global_step": global_step,
                }
                if config.get("save_latest", False):
                    torch.save(save_obj, join(config.output_dir, "ckpt_latest.pth"))
                else:
                    torch.save(save_obj, join(config.output_dir, f"ckpt_{epoch:02d}.pth"))

        with torch.cuda.amp.autocast(enabled=config.use_half_precision, dtype=data_type):
            res = main_with_ensemble(config, test_loader, model_without_ddp, tokenizer, data_type=data_type)
            eval_res = res

        if is_main_process():
            if config.wandb.enable:
                log_dict_to_wandb(eval_res, step=global_step, prefix=config.test_types)

            map = eval_res["map"]   
            logger.info(f"Epoch {epoch}")
            logger.info(f"\n{eval_res}")

            save_json(eval_res, join(config.output_dir, "eval_res_latest.json"))

            if not config.evaluate and map > best:
                if not hasattr(config, "deepspeed") or not config.deepspeed.enable:
                    torch.save(save_obj, join(config.output_dir, "ckpt_best.pth"))
                eval_file = "eval_res_best.json"
                save_json(eval_res, join(config.output_dir, eval_file))
                best = map
                best_epoch = epoch
            if config.evaluate:
                eval_file = "eval_res.json"
                save_json(eval_res, join(config.output_dir, eval_file))

        if hasattr(config, "deepspeed") and config.deepspeed.enable:
            map_best = torch.tensor([0.0, 0.0]).to(device)
            if is_main_process():
                map_best[0] = map
                map_best[1] = best
            dist.broadcast(map_best, 0)
            map, best = map_best[0].item(), map_best[1].item()
        
            if not config.evaluate and map > best:
                model.save_checkpoint(config.output_dir, tag="ckpt_best.pth", save_latest=False, exclude_frozen_parameters=True)

        if config.evaluate or config.debug:
            break

        dist.barrier()

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    logger.info(f"Training time {total_time_str}")
    logger.info(f"best epoch {best_epoch}")
    logger.info(f"best {best}")
    logger.info(f"Checkpoints and Logs saved at {config.output_dir}")

    if is_main_process() and config.wandb.enable:
        run.finish()


def eval_after_training(train_config):
    # general config for all
    train_config.wandb.enable = False
    train_config.evaluate = True
    train_config.pretrained_path = join(train_config.output_dir, "ckpt_best.pth")
    train_config.num_frames_test = train_config.num_frames
    train_config.inputs.video_input.num_frames_test = train_config.num_frames

    if train_config.get('num_frames_test_final', False):
        train_config.num_frames_test = train_config.num_frames_test_final
        train_config.batch_size = train_config.batch_size_final
        train_config.inputs.video_input.num_frames_test = train_config.num_frames_test_final
        train_config.model.vision_encoder.num_frames = train_config.num_frames_test_final

    eval_config = copy.deepcopy(train_config)
    eval_config.test_types = list(eval_config.test_file.keys())
    eval_config.output_dir = join(eval_config.output_dir, f"eval_after_training")
    eval_config.result_dir = eval_config.output_dir
    if is_main_process():
        os.makedirs(eval_config.output_dir, exist_ok=True)
        Config.dump(eval_config, os.path.join(eval_config.output_dir, "config.json"))
    logger.info(f"===========> START eval_after_training [{eval_config.test_types}]")
    main(eval_config)


if __name__ == "__main__":
    cfg = setup_main()
    main(cfg)
    if not cfg.evaluate:
        eval_after_training(cfg)
