import copy
import logging
import os
import os.path as osp
from os.path import join
import io
from copy import deepcopy


import torch
from torch.utils.data import ConcatDataset, DataLoader

from models.backbones.clip.clip_vision import interpolate_pos_embed_vit
from models.backbones.bert.tokenization_bert import BertTokenizer
from utils.optimizer import create_optimizer
from utils.scheduler import create_scheduler
from utils.distributed import get_world_size

import deepspeed

logger = logging.getLogger(__name__)


def get_media_types(datasources):
    """get the media types for for all the dataloaders.

    Args:
        datasources (List): List of dataloaders or datasets.

    Returns: List. The media_types.

    """
    if isinstance(datasources[0], DataLoader):
        datasets = [dataloader.dataset for dataloader in datasources]
    else:
        datasets = datasources
    media_types = [
        dataset.datasets[0].media_type
        if isinstance(dataset, ConcatDataset)
        else dataset.media_type
        for dataset in datasets
    ]

    return media_types


def setup_model(
    config, model_cls, has_decoder=False, pretrain=False, find_unused_parameters=False, num_steps_per_epoch=-1,
):
    logger.info("Creating model")
    config = copy.deepcopy(config)

    # tokenizer = BertTokenizer.from_pretrained(config.model.text_encoder.pretrained)
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased", local_files_only=True)

    model = model_cls(config=config, tokenizer=tokenizer, is_pretrain=pretrain)

    model = model.to(torch.device(config.device))

    model_without_ddp = model

    if hasattr(config, "deepspeed") and config.deepspeed.enable:
        optimizer_params = create_optimizer(config.optimizer, model, return_group=True)
        scheduler = None
        scaler = None
    else:
        if config.distributed:
            model = torch.nn.parallel.DistributedDataParallel(
                model,
                device_ids=[config.gpu],
                find_unused_parameters=find_unused_parameters,  # `False` for image-only task
            )

        optimizer = create_optimizer(config.optimizer, model)
        scaler = torch.cuda.amp.GradScaler(enabled=config.fp16)  # This is never used actually
        scheduler = create_scheduler(config.scheduler, optimizer)

    start_epoch = 0
    global_step = 0

    # auto resume the latest checkpoint
    if config.get("auto_resume", False):
        logger.info("Auto resuming")
        model_latest = join(config.output_dir, "ckpt_latest.pth")
        model_best = join(config.output_dir, "ckpt_best.pth")
        large_num = -1
        for p in os.listdir(config.output_dir):
            if 'ckpt' in p:
                num = p.split('_')[1].split('.')[0]
                if str.isnumeric(num):
                    if int(num) > large_num:
                        large_num = int(num)
        if large_num != -1:
            model_latest = join(config.output_dir, f"ckpt_{large_num:02d}.pth")

        if osp.exists(model_latest):
            config.pretrained_path = model_latest
            config.resume = True
        elif osp.exists(model_best):
            config.pretrained_path = model_best
            config.resume = True
        else:
            logger.info(f"Not found checkpoint in {config.output_dir}")

    if config.pretrained_path.strip() and (osp.isfile(config.pretrained_path) or "s3://" in config.pretrained_path):
        logger.info(f"Loading checkpoint from {config.pretrained_path}")
        
        checkpoint = torch.load(config.pretrained_path, map_location="cpu")
        try:
            state_dict = checkpoint["model"]
        except:  # This is a deepspeed stage 1 model
            state_dict = checkpoint["module"]

        if config.resume:
            optimizer.load_state_dict(checkpoint["optimizer"])
            scheduler.load_state_dict(checkpoint["scheduler"])
            scaler.load_state_dict(checkpoint["scaler"])
            start_epoch = checkpoint["epoch"] + 1
            global_step = checkpoint["global_step"]

        elif config.evaluate or (not pretrain):  # downstream init from pretrained ckpt
            is_blip_model = "VindLU_BLIP" in config.model.get("model_cls", "")

            # interpolate positional embeddings.
            if "vit" in config.model.vision_encoder.name:
                state_dict = interpolate_pos_embed_vit(state_dict, model_without_ddp)
            else:
                raise ValueError(
                    f" vision encoder: {config.model.vision_encoder.name} not implelented"
                )
            if not config.evaluate or config.get("zero_shot", False):  # finetuning from a pretarined weights.
                for key in list(state_dict.keys()):
                    if "bert" in key and not is_blip_model:
                        encoder_key = key.replace("bert.", "")
                        state_dict[encoder_key] = state_dict[key]
                        if not has_decoder:
                            del state_dict[key]

                    # init text decoder as multimodal encoder (last 6 layers of model.text_encoder)
                    # only for generation tasks like VQA
                    if has_decoder and "text_encoder" in key and not is_blip_model:
                        if "layer" in key:
                            encoder_keys = key.split(".")
                            layer_num = int(encoder_keys[4])
                            if layer_num < config.model.text_encoder.fusion_layer:
                                del state_dict[key]
                                continue
                            else:
                                decoder_layer_num = layer_num - config.model.text_encoder.fusion_layer
                                encoder_keys[4] = str(decoder_layer_num)
                                encoder_key = ".".join(encoder_keys)
                        else:
                            encoder_key = key
                        decoder_key = encoder_key.replace("text_encoder", "text_decoder")
                        state_dict[decoder_key] = state_dict[key]
                        del state_dict[key]

        if hasattr(config, "wiseft") and config.wiseft.enable:
            logger.info(f"Wiseft with coefficient {config.wiseft.coef}")
            missing_keys_in_pretrained = [k for k in model_without_ddp.state_dict().keys() if k not in state_dict]
            missing_keys_in_model = [k for k in state_dict.keys() if k not in model_without_ddp.state_dict()]
            mismatch_keys = [k for k in state_dict.keys() if k in model_without_ddp.state_dict() \
                             and state_dict[k].shape != model_without_ddp.state_dict()[k].shape]
            common_keys = [k for k in state_dict.keys() if k in model_without_ddp.state_dict()]
            logger.info(f"Missing keys in pretrained: {missing_keys_in_pretrained}")
            logger.info(f"Missing keys in model: {missing_keys_in_model}")
            logger.info(f"Mismatch keys: {mismatch_keys}")
            logger.info(f"Keys to exclude: {config.wiseft.keys_to_exclude}")
            for k in common_keys:
                if k in config.wiseft.keys_to_exclude:
                    continue
                state_dict[k] = config.wiseft.coef * state_dict[k] + (1 - config.wiseft.coef) * model_without_ddp.state_dict()[k].cpu()

        msg = model_without_ddp.load_state_dict(state_dict, strict=False)
        logger.info(msg)
        logger.info(f"Loaded checkpoint from {config.pretrained_path}")
    else:
        logger.warning("No pretrained checkpoint provided, training from scratch")

    if scaler is None:
        model = model_without_ddp
        model, optimizer, _, _ = deepspeed.initialize(
            args=config, model=model, model_parameters=optimizer_params, dist_init_required=not config.distributed,
            lr_scheduler=lambda opt: create_scheduler(config.scheduler, opt)
        )
    
    if config.resume and osp.isdir(config.pretrained_path):
        output_dir, tag = os.path.split(config.pretrained_path)
        model.load_checkpoint(output_dir, tag=tag)
        global_step = model.global_steps
        assert num_steps_per_epoch > 0, "Please provide num_steps_per_epoch"
        start_epoch = global_step // num_steps_per_epoch

    return (
        model,
        model_without_ddp,
        optimizer,
        scheduler,
        scaler,
        tokenizer,
        start_epoch,
        global_step,
    )
