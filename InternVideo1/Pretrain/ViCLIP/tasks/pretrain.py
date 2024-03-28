import datetime
import logging
import time
from os.path import join

import pandas as pd
import torch
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import wandb
from torch.utils.data import ConcatDataset

from dataset import MetaLoader, create_dataset, create_loader, create_sampler
from dataset.serialize import local_broadcast_process_authkey

from models.viclip import ViCLIP

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
    metric_logger.add_meter("lr", SmoothedValue(window=100, fmt="{value:.6f}"))
    metric_logger.add_meter("temperature", SmoothedValue(window=100, fmt="{value:.4f}"))
    loss_names = ["loss_" + k for k, v in config.criterion.loss_weight.items() if v != 0]
    requires_raw_text = config.criterion.get('mac_all', False) or \
        config.model.get("requires_raw_text", False)

    media_types = get_media_types(train_loaders)

    for name in loss_names:
        for m in media_types:
            metric_logger.add_meter(
                f"{m}-{name}", SmoothedValue(window=100, fmt="{value:.4f}")
            )

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
            max_length=config.inputs.max_txt_l[media_type],
            return_tensors="pt",
        ).to(
            device
        )  # change from "longest" to "max_length"

        with torch.cuda.amp.autocast(enabled=config.fp16, dtype=torch.bfloat16):
            if requires_raw_text:
                loss_dict = model(image, text_input, idx=idx, raw_text=text, log_generation=(i % log_freq == 0))
            else:
                loss_dict = model(image, text_input, idx=idx)
            loss = sum(loss_dict.values())

        if hasattr(config, "deepspeed") and config.deepspeed.enable:
            model.backward(loss)
            model.step()
        else:  #! We do not use scaler as we only involve bf16, check this
            optimizer.zero_grad()
            loss.backward()
            if config.optimizer.max_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), config.optimizer.max_grad_norm)
            optimizer.step()
            scheduler.step()

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

        if config.debug and global_step % 20 == 0:
            logger.info("debug mode, break training loop")
            break

        if config.debug and global_step % (2 * log_freq + 3) == 0:
            logger.info("debug mode, break training loop")
            break

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
    )
    
    # test datasets, a mapping from dataset name to data loader
    test_datasets, test_dataset_names = create_dataset(f"{mode}_eval", config)
    test_samplers = []
    for test_dataset, test_dataset_name in zip(test_datasets, test_dataset_names):
        if "_qa_" in test_dataset_name and config.distributed:
            test_samplers.append(
                create_sampler([test_dataset], [False], num_tasks, global_rank)[0]
            )
        else:
            test_samplers.append(None)
    test_loaders = create_loader(
        test_datasets,
        # [None] * len(test_datasets),
        test_samplers,
        batch_size=[config.inputs.batch_size_test[d.media_type] for d in test_datasets],
        num_workers=[config.num_workers] * len(test_datasets),
        is_trains=[False] * len(test_datasets),
        collate_fns=[None] * len(test_datasets),
    )
    test_name2loaders = {k: v for k, v in zip(test_dataset_names, test_loaders)}
    return train_loaders, test_name2loaders, media_types


def main(config):
    if is_main_process() and config.wandb.enable:
        run = setup_wandb(config)

    is_pretrain = config.mode == "pt"

    logger.info(f"train_file: {config.train_file}")

    setup_seed(config.seed + get_rank())
    device = torch.device(config.device)

    train_loaders, test_name2loaders, train_media_types = setup_dataloaders(
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
            )
        with torch.cuda.amp.autocast(enabled=config.fp16, dtype=torch.bfloat16):
            eval_res = {}
            for test_name, test_loader in test_name2loaders.items():
                if test_name not in config.test_types:
                    logger.info(
                        f"Skip eval {test_name} split. All test_types {config.test_types}"
                    )
                    continue
                if not "_qa_" in test_name:  # Retrieval test
                    res = ret_eval_wrapper(
                        model_without_ddp, test_loader, tokenizer, device, config, prefix=test_name
                    )
                else:
                    res = qa_eval_wrapper(
                        model, test_loader, tokenizer, device, config, prefix=test_name
                    )
                eval_res.update(res)
        
        if len(eval_res) == 0:
            logger.info("Evaluation results are empty, using fake results")
            eval_res = {"msrvtt_1k_test\/":{"txt_r1":0.0,"txt_r5":0.0,"txt_r10":0.0,"txt_r_mean":0.0,"img_r1":0.0,"img_r5":0.0,"img_r10":0.0,"img_r_mean":0.0,"r_mean":0.0},
                        "msrvtt_1k_test_emb\/":{"txt_r1":0.0,"txt_r5":0.0,"txt_r10":0.0,"txt_r_mean":0.0,"img_r1":0.0,"img_r5":0.0,"img_r10":0.0,"img_r_mean":0.0,"r_mean":0.0}}

        if is_main_process():

            # log to wandb
            if config.wandb.enable:
                for p, v in eval_res.items():
                    log_dict_to_wandb(v, step=global_step, prefix=p)

            if config.stop_key is not None and config.stop_key in eval_res:
                if config.model.multimodal.enable:
                    cur_r_mean = eval_res[config.stop_key]["r_mean"]
                else:
                    cur_r_mean = eval_res[config.stop_key.replace("/", "_emb/")]["r_mean"]
            else:  # None
                cur_r_mean = best + 1  # save the last as the best

            eval_res = pd.DataFrame(eval_res)
            logger.info(f"Epoch {epoch}")
            logger.info(f"\n{eval_res.transpose().to_string(max_cols=30)}")

            eval_res.to_json(join(config.output_dir, "eval_res_latest.json"))

            state_dict = model_without_ddp.state_dict()

            for k in config.get("no_save_params_prefix", []):
                kk = [x for x in state_dict.keys() if x.startswith(k)]
                logger.info(f"Not saving {len(kk)} params with prefix {k}")
                for kkk in kk:
                    state_dict.pop(kkk)
            
            if scaler is not None:
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

            if not config.evaluate and cur_r_mean > best:
                if scaler is not None:
                    torch.save(save_obj, join(config.output_dir, "ckpt_best.pth"))
                eval_file = "eval_res_best.json"
                eval_res.to_json(join(config.output_dir, eval_file))
                best = cur_r_mean
                best_epoch = epoch
        
        r_mean_best = torch.tensor([0.0, 0.0]).to(device)
        if is_main_process():
            r_mean_best[0] = cur_r_mean
            r_mean_best[1] = best
        dist.broadcast(r_mean_best, 0)
        cur_r_mean, best = r_mean_best[0].item(), r_mean_best[1].item()
        
        if scaler is None:  # deepspeed
            if config.get("save_latest", False):
                tag = "ckpt_latest.pth"
            else:
                tag = f"ckpt_{epoch:02d}.pth"
        
            model.save_checkpoint(config.output_dir, tag=tag, save_latest=False)
            if not config.evaluate and cur_r_mean > best:
                model.save_checkpoint(config.output_dir, tag="ckpt_best.pth", save_latest=False)

        if config.evaluate:
            break

        dist.barrier()

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    logger.info(f"Training time {total_time_str}")
    logger.info(f"best epoch {best_epoch} [config.stop_key {config.stop_key}]")
    logger.info(f"Checkpoints and Logs saved at {config.output_dir}")

    if is_main_process() and config.wandb.enable:
        run.finish()


if __name__ == "__main__":
    cfg = setup_main()
    local_broadcast_process_authkey()
    main(cfg)
