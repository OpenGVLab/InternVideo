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

from dataset import MetaLoader
from models.vindlu import VindLU
from models.vindlu_vit import VindLU_VIT
from models.vindlu_videoclip import VindLU_VideoCLIP
from models.vindlu_blip_qformer import VindLU_BLIP_QFormer
from tasks.pretrain import setup_dataloaders
from tasks.retrieval_utils import evaluation_wrapper
from tasks.shared_utils import setup_model
from utils.basic_utils import MetricLogger, SmoothedValue, setup_seed
from utils.config import Config
from utils.config_utils import setup_main
from utils.distributed import get_rank, is_main_process
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
    metric_logger.add_meter("lr", SmoothedValue(window=1, fmt="{value:.6f}"))
    metric_logger.add_meter("temperature", SmoothedValue(window=1, fmt="{value:.4f}"))
    loss_names = ["loss_" + k for k, v in config.criterion.loss_weight.items() if v != 0]
    requires_raw_text = config.model.get("requires_raw_text", False)

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

        with torch.cuda.amp.autocast(enabled=config.fp16, dtype=torch.bfloat16):
            if requires_raw_text:
                loss_dict = model(image, text_input, idx=idx, raw_text=text)
            else:
                loss_dict = model(image, text_input, idx=idx)
            loss = sum(loss_dict.values())
        
        #! We do not use scaler as we only involve bf16, check this
        optimizer.zero_grad()
        loss.backward()
        if config.optimizer.max_grad_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.optimizer.max_grad_norm)
        optimizer.step()
        scheduler.step()

        # optimizer.zero_grad()
        # scaler.scale(loss).backward()
        # if config.optimizer.max_grad_norm > 0:
        #     scaler.unscale_(optimizer)
        #     torch.nn.utils.clip_grad_norm_(model.parameters(), config.optimizer.max_grad_norm)
        # scaler.step(optimizer)
        # scaler.update()
        # scheduler.step()

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

    train_loaders, test_name2loaders, train_media_types = setup_dataloaders(config, mode="ret")
    num_steps_per_epoch = sum(len(d) for d in train_loaders)
    config.scheduler.num_training_steps = num_steps_per_epoch * config.scheduler.epochs
    config.scheduler.num_warmup_steps = num_steps_per_epoch * config.scheduler.warmup_epochs

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
        pretrain=False,
        # find_unused_parameters=True
        find_unused_parameters=find_unused_parameters,
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

        if config.get('no_test', False) and not config.evaluate:
            save_obj = {
                "model": model_without_ddp.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "scaler": scaler.state_dict(),
                "config": config,
                "epoch": epoch,
                "global_step": global_step,
            }
            torch.save(save_obj, join(config.output_dir, "ckpt_best.pth"))
            best_epoch = epoch

        else:
            with torch.cuda.amp.autocast(enabled=config.fp16, dtype=torch.bfloat16):
                eval_res = {}
                for test_name, test_loader in test_name2loaders.items():
                    if test_name not in config.test_types:
                        logger.info(
                            f"Skip eval {test_name} split. All test_types {config.test_types}"
                        )
                        continue
                    res = evaluation_wrapper(
                        model_without_ddp, test_loader, tokenizer, device, config, prefix=test_name
                    )
                    eval_res.update(res)

            if is_main_process():
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

                if not config.evaluate and cur_r_mean > best:
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
                    eval_res.to_json(join(config.output_dir, eval_file))
                    torch.save(save_obj, join(config.output_dir, "ckpt_best.pth"))
                    best = cur_r_mean
                    best_epoch = epoch
                if config.evaluate:
                    eval_file = "eval_res.json"
                    eval_res.to_json(join(config.output_dir, eval_file))

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
