import datetime
import json
import logging
import os.path as osp
import time

import pandas as pd
import torch
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import wandb

from dataset import MetaLoader, create_dataset, create_loader, create_sampler
from models.vindlu import VindLU
from tasks.retrieval_utils import evaluation_wrapper
from tasks.shared_utils import get_media_types, setup_model
from utils.basic_utils import (MetricLogger, SmoothedValue,
                               remove_files_if_exist, setup_seed)
from utils.config_utils import setup_main
from utils.distributed import get_rank, get_world_size, is_main_process
from utils.logger import log_dict_to_wandb, setup_wandb

logger = logging.getLogger(__name__)


class PretrainTrainer(object):
    """trainer for pretraining."""

    def __init__(self, config):
        super(PretrainTrainer, self).__init__()
        self.config = config

        self.is_pretrain = config.mode == "pt"
        self.setup()

        self.has_decoder = False
        if config.mode in ["ret", "pt"]:
            self.evaluation_fn = evaluation_wrapper
            self.model_cls = VindLU
        elif config.mode == "vqa":
            raise NotImplementedError("not implemented")
        else:
            raise NotImplementedError("not implemented")

        self.build_dataloaders()
        self.build_model()

    def setup(self):
        """setup for train."""
        config = self.config
        if is_main_process() and config.wandb.enable:
            self.wandb_run = setup_wandb(config)
        else:
            self.wandb_run = None
        setup_seed(config.seed + get_rank())
        self.device = torch.device(config.device)

    @torch.no_grad()
    def evaluate(self, epoch=0):
        """evaluate the model.
        Args:
            model (nn.Module): The model to evaluate.
            loader (DataLoader): dataloader.
            tokenizer (None): tokenizer.
            prefix (str): The str prepended to the keys of return dict.

        Returns: dict. The value is the corresponding evaluation results for the key.
        """
        eval_res = {}
        for test_name, test_loader in self.test_name2loaders.items():
            if test_name not in self.config.test_types:
                logger.info(
                    f"Skip eval {test_name} split. All test_types {self.config.test_types}"
                )
                continue
            with torch.cuda.amp.autocast(enabled=self.config.fp16):
                res = self.evaluation_fn(
                    self.model_without_ddp,
                    test_loader,
                    self.tokenizer,
                    self.device,
                    self.config,
                    test_name,
                )
            eval_res.update(res)

        df = pd.DataFrame(eval_res)
        logger.info(f"Epoch {epoch}")
        logger.info(f"\n{df.transpose().to_string(max_cols=30)}")
        return eval_res

    def build_model(self):
        """TODO: Docstring for build_model.
        Returns: TODO

        """
        (
            self.model,
            self.model_without_ddp,
            self.optimizer,
            self.scheduler,
            self.scaler,
            self.tokenizer,
            self.start_epoch,
            self.global_step,
        ) = setup_model(
            self.config,
            model_cls=self.model_cls,
            has_decoder=self.has_decoder,
            pretrain=self.is_pretrain,
            find_unused_parameters=True,
        )

    def build_dataloaders(self):
        config = self.config
        mode = config.mode
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

        # test datasets, a mapping from dataset name to data loader
        test_datasets, test_dataset_names = create_dataset(f"{mode}_eval", config)
        test_loaders = create_loader(
            test_datasets,
            [None] * len(test_datasets),
            batch_size=[config.inputs.batch_size_test[d.media_type] for d in test_datasets],
            num_workers=[config.num_workers] * len(test_datasets),
            is_trains=[False] * len(test_datasets),
            collate_fns=[None] * len(test_datasets),
        )
        test_name2loaders = {k: v for k, v in zip(test_dataset_names, test_loaders)}

        self.train_loaders = train_loaders
        self.test_name2loaders = test_name2loaders
        self.media_types = media_types

        num_steps_per_epoch = sum(len(d) for d in self.train_loaders)
        # update config
        config.scheduler.num_training_steps = num_steps_per_epoch * config.scheduler.epochs
        config.scheduler.num_warmup_steps = (
            num_steps_per_epoch * config.scheduler.warmup_epochs
        )
        self.config = config

    def train(self):
        """train the model."""
        config = self.config

        # set cudnn.benchmark=True only when input size is fixed
        # https://discuss.pytorch.org/t/what-does-torch-backends-cudnn-benchmark-do/5936/3
        cudnn.benchmark = len(self.media_types) == 1

        if is_main_process() and config.wandb.enable:
            wandb.watch(self.model)

        best = 0
        best_epoch = 0

        logger.info("Start training")
        start_time = time.time()
        global_step = self.global_step
        for epoch in range(self.start_epoch, config.scheduler.epochs):
            # train one epoch
            global_step = self.train_one_epoch(epoch, global_step)

            # evaluation.
            eval_res = self.evaluate(epoch)

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

                with open(osp.join(config.output_dir, "eval_res_latest.json"), "w") as f:
                    json.dump(eval_res, f)
                # eval_res.to_json(osp.join(config.output_dir, "eval_res_latest.json"))

                save_obj = {
                    "model": self.model_without_ddp.state_dict(),
                    "optimizer": self.optimizer.state_dict(),
                    "scheduler": self.scheduler.state_dict(),
                    "scaler": self.scaler.state_dict(),
                    "config": config,
                    "epoch": epoch,
                    "global_step": global_step,
                }
                torch.save(save_obj, osp.join(config.output_dir, f"ckpt_{epoch:02d}.pth"))

                if cur_r_mean > best:
                    torch.save(save_obj, osp.join(config.output_dir, "ckpt_best.pth"))
                    eval_file = "eval_res_best.json"
                    # eval_res.to_json(osp.join(config.output_dir, eval_file))
                    with open(osp.join(config.output_dir, eval_file), "w") as f:
                        json.dump(eval_res, f)
                    best = cur_r_mean
                    best_epoch = epoch

            dist.barrier()

        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        logger.info(f"Training time {total_time_str}")
        logger.info(f"best epoch {best_epoch} [config.stop_key {config.stop_key}]")
        logger.info(f"Checkpoints and Logs saved at {config.output_dir}")

        if is_main_process() and config.wandb.enable:
            self.wandb_run.finish()

    def train_one_epoch(self, epoch, global_step):
        config = self.config
        self.model.train()

        metric_logger = MetricLogger(delimiter="  ")
        metric_logger.add_meter("lr", SmoothedValue(window=100, fmt="{value:.6f}"))
        metric_logger.add_meter("temperature", SmoothedValue(window=100, fmt="{value:.4f}"))
        loss_names = ["loss_" + k for k, v in config.criterion.loss_weight.items() if v != 0]

        media_types = get_media_types(self.train_loaders)

        for name in loss_names:
            for m in media_types:
                metric_logger.add_meter(
                    f"{m}-{name}", SmoothedValue(window=100, fmt="{value:.4f}")
                )

        header = f"Train Epoch: [{epoch}]"
        log_freq = config.log_freq

        if config.distributed:
            for d in self.train_loaders:
                d.sampler.set_epoch(epoch)
        train_loader = MetaLoader(name2loader=dict(list(zip(media_types, self.train_loaders))))

        model_without_ddp = self.model.module if config.distributed else self.model
        iterator = metric_logger.log_every(train_loader, log_freq, header)
        for i, (media_type, (image, text, idx)) in enumerate(iterator):
            image = image.to(self.device, non_blocking=True)
            idx = idx.to(self.device, non_blocking=True)
            text_input = self.tokenizer(
                text,
                padding="max_length",
                truncation=True,
                max_length=config.inputs.max_txt_l[media_type],
                return_tensors="pt",
            ).to(self.device)

            with torch.cuda.amp.autocast(enabled=config.fp16, dtype=torch.bfloat16):
                loss_dict = self.model(image, text_input, idx=idx)
                loss = sum(loss_dict.values())

            self.optimizer.zero_grad()
            self.scaler.scale(loss).backward()
            if config.optimizer.max_grad_norm > 0:
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), config.optimizer.max_grad_norm
                )
            self.scaler.step(self.optimizer)
            self.scaler.update()
            self.scheduler.step()

            # logging
            for name in loss_names:
                value = loss_dict[name]
                value = value if isinstance(value, float) else value.item()
                metric_logger.update(**{f"{media_type}-{name}": value})
            metric_logger.update(lr=self.optimizer.param_groups[0]["lr"])
            metric_logger.update(temperature=model_without_ddp.temp.item())

            if is_main_process() and config.wandb.enable and global_step % log_freq == 0:
                logs = metric_logger.get_global_avg_dict()
                log_dict_to_wandb(logs, step=global_step, prefix="train/")

            global_step += 1

            if config.debug and global_step % 2 == 0:
                logger.info("debug mode, break training loop")
                break

        # gather the stats from all processes
        metric_logger.synchronize_between_processes()
        logger.info(f"Averaged stats: {metric_logger.global_avg()}")
        return global_step


if __name__ == "__main__":
    cfg = setup_main()
    trainer = PretrainTrainer(cfg)
    if cfg.evaluate:
        trainer.evaluate()
    else:
        trainer.train()
