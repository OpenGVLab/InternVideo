import copy
import logging
import os
import os.path as osp
import io

try:
    import deepspeed
except Exception as e:
    print(e)
    print("deepspeed is not installed!!!")
    
from os.path import join


try:
    from petrel_client.client import Client
except:
    Client = None

import torch
from torch.utils.data import ConcatDataset, DataLoader

from dataset.resample_concat_dataset import ResampleConcatDataset
from models.backbones.internvideo2.pos_embed import interpolate_pos_embed_internvideo2_new
from models.backbones.bert.tokenization_bert import BertTokenizer
from utils.optimizer import create_optimizer
from utils.scheduler import create_scheduler
from utils.distributed import get_rank

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
        if isinstance(dataset, ConcatDataset) or isinstance(dataset, ResampleConcatDataset)
        else dataset.media_type
        for dataset in datasets
    ]

    return media_types


def setup_model(
    config, model_cls, add_decoder=False, pretrain=False, find_unused_parameters=False
):
    logger.info("Creating model")
    config = copy.deepcopy(config)

    if "bert" in config.model.text_encoder.name:
        logger.info(f"Using BertTokenizer: {config.model.text_encoder.pretrained}!")
        tokenizer = BertTokenizer.from_pretrained(config.model.text_encoder.pretrained, local_files_only=True)
        model = model_cls(config=config, tokenizer=tokenizer, is_pretrain=pretrain)
    else:
        model = model_cls(config=config, is_pretrain=pretrain)
        tokenizer = model.tokenizer
        logger.info(f"Using model.tokenizer: {tokenizer}!")

    
    if config.get('compile_model', False):
        torch.set_float32_matmul_precision('high')
        model = torch.compile(model)

    model = model.to(torch.device(config.device))
    model_without_ddp = model


    if hasattr(config, "deepspeed") and config.deepspeed.enable:
        # We move this to the back
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
        scaler = torch.cuda.amp.GradScaler(enabled=config.use_half_precision)  # This is never used actually if we fixed bf16
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
        if osp.isfile(model_latest):
            config.pretrained_path = model_latest
            config.resume = True
        elif osp.isfile(model_best):
            config.pretrained_path = model_best
            config.resume = True
        else:
            logger.info(f"Not found checkpoint in {config.output_dir}")


    if (config.pretrained_path.strip() and (osp.isfile(config.pretrained_path)) or "s3://" in config.pretrained_path):
        if Client is not None:
            client = Client()
            with io.BytesIO(client.get(config.pretrained_path)) as buffer:
                checkpoint = torch.load(buffer, map_location="cpu")
        else:
            checkpoint = torch.load(config.pretrained_path, map_location="cpu")
        logger.info(f"Loading checkpoint from {config.pretrained_path}")
        try:
            if "model" in checkpoint.keys():
                state_dict = checkpoint["model"]
            else:
                state_dict = checkpoint["module"] # This is a deepspeed stage 1 model
        except:  
            state_dict = checkpoint

        if config.get('origin_num_frames', None) is not None:
            logger.info(f"interpolate_pos_embed_internvideo2 (origin_num_frames={config.origin_num_frames})!!!")
            a = len(state_dict)
            interpolate_pos_embed_internvideo2_new(state_dict, model_without_ddp.vision_encoder, orig_t_size=config.origin_num_frames)
            assert a == len(state_dict), state_dict.keys()

        if config.resume:
            assert not (hasattr(config, "deepspeed") and config.deepspeed.enable), "Deepspeed should run here!!!"
            optimizer.load_state_dict(checkpoint["optimizer"])
            scheduler.load_state_dict(checkpoint["scheduler"])
            scaler.load_state_dict(checkpoint["scaler"])
            if 'local_step' in checkpoint.keys():
                start_epoch = checkpoint['epoch']
            else:
                start_epoch = checkpoint["epoch"] + 1
            global_step = checkpoint["global_step"]

        elif not pretrain:  # downstream init from pretrained ckpt 


            if not config.evaluate or config.get("zero_shot", False):  # finetuning from a pretrained weights.
                if add_decoder:
                    logger.info("Init new decoder with encoder!!!")
                for key in list(state_dict.keys()):
                    if "text_encoder.bert" in key:
                        encoder_key = key.replace("bert.", "") 
                        state_dict[encoder_key] = state_dict[key]
                        if not add_decoder:
                            del state_dict[key]

                    # init text decoder as multimodal encoder (last 6 layers of model.text_encoder)
                    # only for generation tasks like VQA
                    if add_decoder and "text_encoder.bert" in key:
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


        msg = model_without_ddp.load_state_dict(state_dict, strict=False)
        logger.info(msg)
        logger.info(f"Loaded checkpoint from {config.pretrained_path}")
    else:
        if not config.resume:
            assert not config.evaluate, "No available pretrained checkpoint provided!!!"
            assert config.pretrained_path == "", config.pretrained_path
            logger.warning("No available pretrained checkpoint provided, training from scratch.")


    if hasattr(config, "deepspeed") and config.deepspeed.enable:
        logger.info(f'Use deepspeed to initialize model (resume={config.resume}) !!!')
        model = model_without_ddp

        model, optimizer, _, _ = deepspeed.initialize(
            args=config, model=model, model_parameters=optimizer_params, dist_init_required=not config.distributed,
            lr_scheduler=lambda opt: create_scheduler(config.scheduler, opt)
        )


        if config.resume:
            logger.info(f'Resume deepspeed ckpt from {config.output_dir}, tag={config.pretrained_path}, load_module_strict={config.get("load_module_strict", True)}, load_lr_scheduler_states={config.get("load_lr_scheduler_states", True)}!!!')
            _, client_states = model.load_checkpoint(config.output_dir, tag=config.pretrained_path,  load_module_strict=config.get("load_module_strict", True), load_lr_scheduler_states=config.get("load_lr_scheduler_states", True))
            logger.info(client_states)
            if 'local_step' in client_states.keys():
                start_epoch = client_states['epoch']
            else:
                start_epoch = client_states['epoch'] + 1
            global_step = client_states['global_step']


    logger.info(f"Cuda memory after create model: {torch.cuda.memory_allocated() // 1024**2}M, Max mem: {torch.cuda.max_memory_allocated() // 1024**2}M start_epoch={start_epoch}, global_step={global_step}")
    print(f"\033[31m Cuda memory after create model: {torch.cuda.memory_allocated() // 1024**2}M, Max mem: {torch.cuda.max_memory_allocated() // 1024**2}M start_epoch={start_epoch}, global_step={global_step}\033[0m")

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
