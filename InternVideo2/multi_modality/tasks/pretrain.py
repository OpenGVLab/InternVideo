import datetime
import logging
import time
from os.path import join

import pandas as pd
import torch
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import wandb

from dataset.serialize import local_broadcast_process_authkey
from dataset import MetaLoader, MetaLoader_rs2, create_dataset, create_loader, create_sampler, create_stateful_sampler
from models import *
from tasks.retrieval_utils import evaluation_wrapper
from tasks.shared_utils import get_media_types, setup_model
from utils.basic_utils import (MetricLogger, SmoothedValue,
                               remove_files_if_exist, setup_seed)
from utils.config_utils import setup_main
from utils.distributed import get_rank, get_world_size, is_main_process
from utils.logger import log_dict_to_wandb, setup_wandb
try:
    from petrel_client.client import Client
except:
    Client = None
import io
import os
import shutil

logger = logging.getLogger(__name__)

ceph_ckpt_bucket = "shdd:s3://avp_ckpt"


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
    skip_num=0
):

    try:
        ceph_ckpt_path = f"{ceph_ckpt_bucket}/{config.output_dir.split('/')[-3]}/{config.output_dir.split('/')[-2]}/{config.output_dir.split('/')[-1]}"
        client_ckpt = Client(conf_path='~/petreloss.conf')
    except Exception as e:
        print(e)
        logger.info("Ceph is not working!!!")


    if config.use_half_precision: 
        if config.get('use_bf16', False):
            cast_dtype = torch.bfloat16
        else:
            cast_dtype = torch.float16
    else:
        cast_dtype = None

    model.train()

    metric_logger = MetricLogger(delimiter="  ")
    metric_logger.add_meter("lr", SmoothedValue(window=100, fmt="{value:.6f}"))
    metric_logger.add_meter("temperature", SmoothedValue(window=100, fmt="{value:.4f}"))
    loss_names = ["loss_" + k for k, v in config.criterion.loss_weight.items() if v != 0]
    
    if config.get("use_raw_text", False): # for cosa
        loss_names = loss_names + ["c_loss_" + k for k, v in config.criterion.loss_weight.items() if v != 0]
    uta_all = config.criterion.get('uta_all', False)

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

    if config.get('use_iter_train', False):
        train_loader = MetaLoader_rs2(name2loader=dict(list(zip(media_types, train_loaders))), skip_num=skip_num)
    else:
        train_loader = MetaLoader(name2loader=dict(list(zip(media_types, train_loaders))))

    model_without_ddp = model.module if config.distributed else model
    iterator = metric_logger.log_every(train_loader, log_freq, header)
    
    begin_step = global_step % len(train_loader)
    logger.info(f"Epoch={epoch}, begin_step={begin_step} save_ckpt_iter={config.get('save_ckpt_iter', None)}")

    for local_step, (media_type, (media, text, idx)) in enumerate(iterator):
        if local_step < begin_step: 
            logger.warn(f"Jump local_step: {local_step} (begin_step={begin_step})!!!")
            metric_logger.update(lr=optimizer.param_groups[0]["lr"])
            continue

        if config.get("save_ckpt_iter", None) is not None: # and not is_iter_resume:
            if local_step != 0 and local_step % config.get("save_ckpt_iter") == 0:
                if hasattr(config, "deepspeed") and config.deepspeed.enable: 
                    tag = f"ckpt_e{epoch:02d}_local{local_step}_global{global_step}"
                    client_state = {'epoch': epoch, 'global_step': global_step, 'local_step': local_step}
                    model.save_checkpoint(config.output_dir, tag=tag, save_latest=False, client_state=client_state)
                    logger.info(f"save ckpt file to local ({config.output_dir}/{tag})!!!")
                elif is_main_process():
                    state_dict = model_without_ddp.state_dict()
                    for k in config.get("no_save_params_prefix", []):
                        kk = [x for x in state_dict.keys() if x.startswith(k)]
                        logger.info(f"Not saving {len(kk)} params with prefix {k}")
                        for kkk in kk:
                            state_dict.pop(kkk)

                    save_obj = {
                        "model": state_dict,
                        "optimizer": optimizer.state_dict(),
                        "scheduler": scheduler.state_dict(),
                        "scaler": scaler.state_dict(),
                        "config": config,
                        "epoch": epoch,
                        "local_step": local_step,
                        "global_step": global_step,
                    }
                    try:
                        with io.BytesIO() as buffer:
                            torch.save(save_obj, buffer)
                            client_ckpt.put(f"{ceph_ckpt_path}/ckpt_{epoch:02d}_local{local_step}_global{global_step}.pth", buffer.getvalue())
                            logger.info(f"Save to ceph ({ceph_ckpt_path}/ckpt_{epoch:02d}_local{local_step}_global{global_step}.pth)!!!")
                    except Exception as e:
                        print(e)
                        torch.save(save_obj, join(config.output_dir, f"ckpt_{epoch:02d}_local{local_step}_global{global_step}.pth"))
                        logger.warn(f"Ceph is not working, save to local ({join(config.output_dir, f'ckpt_{epoch:02d}_local{local_step}_global{global_step}.pth')})!!!")

        if media_type == 'audio_video':
            if type(media[0]) is list:
                assert len(media[0]) == 2
                audio = [media[0][0].to(device, dtype=cast_dtype, non_blocking=True), media[0][1].to(device, non_blocking=True)]
            else:
                audio = media[0].to(device, dtype=cast_dtype, non_blocking=True)
            video = media[1].to(device, dtype=cast_dtype, non_blocking=True)
            media = [audio, video]
        else:
            media = media.to(device, dtype=cast_dtype, non_blocking=True)
        idx = idx.to(device, non_blocking=True)
        if config.get("use_raw_text", False) or config.get("use_cosa", False):
            max_length = config.inputs.max_txt_l[media_type]
        else:
            if type(text) is dict:
                text_input = {}
                for k in text.keys():
                    text_input[k] = tokenizer(
                    text[k],
                    padding="max_length",
                    truncation=True,
                    max_length=config.inputs.max_txt_l[media_type],
                    return_tensors="pt",
                ).to(
                    device) # change from "longest" to "max_length"
            else:
                text_input = tokenizer(
                    text,
                    padding="max_length",
                    truncation=True,
                    max_length=config.inputs.max_txt_l[media_type],
                    return_tensors="pt",
                ).to(
                    device) # change from "longest" to "max_length"
            

        if hasattr(config, "deepspeed") and config.deepspeed.enable:
            loss_dict = model(media, text_input, idx=idx, media_type=media_type)
            loss = sum(loss_dict.values())

            model.backward(loss)
            model.step()

        else: # NOTE We shouldn't use scaler if we only involve bf16, check this!
            with torch.cuda.amp.autocast(enabled=config.use_half_precision, dtype=cast_dtype):
                loss_dict = model(media, text_input, idx=idx, media_type=media_type)
                loss = sum(loss_dict.values())

            if not config.use_half_precision or config.get('use_bf16', False):
                optimizer.zero_grad()
                loss.backward()
                if config.optimizer.max_grad_norm > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), config.optimizer.max_grad_norm)
                    
                optimizer.step()
                scheduler.step()
            else:
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
            if name in loss_dict.keys():
                value = loss_dict[name]
                value = value if isinstance(value, float) else value.item()
                metric_logger.update(**{f"{media_type}-{name}": value})
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])
        metric_logger.update(temperature=model_without_ddp.temp.item())

        if is_main_process() and config.wandb.enable and global_step % log_freq == 0:
            try:
                logs = metric_logger.get_global_avg_dict()
                log_dict_to_wandb(logs, step=global_step, prefix="train/")
            except Exception as e:
                logger.warn("Wandb is not working!!!")
                print(e)
            
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
    logger.info(f"Creating dataset for {mode} use_iter_train={config.get('use_iter_train', False)}")
    train_datasets = create_dataset(f"{mode}_train", config)
    media_types = get_media_types(train_datasets)

    if config.get('use_iter_train', False):
        if config.distributed:
            batch_size = [config.inputs.batch_size[k] for k in media_types] # batch_size for each GPU
            samplers = create_stateful_sampler(train_datasets, batch_size)
        else:
            raise NotImplementedError
    else:
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
    return train_loaders, test_name2loaders, media_types


def main(config):
    if config.get('use_flash_sdp', False):
        torch.backends.cuda.enable_flash_sdp(enabled=True)
    elif config.get('use_mem_efficient_sdp', False):
        torch.backends.cuda.enable_mem_efficient_sdp(enabled=True)

    try:
        ceph_ckpt_path = f"{ceph_ckpt_bucket}/{config.output_dir.split('/')[-3]}/{config.output_dir.split('/')[-2]}/{config.output_dir.split('/')[-1]}"
        client_ckpt = Client(conf_path='~/petreloss.conf')
    except Exception as e:
        print(e)
        logger.info("Ceph is not working!!!")
        
    if is_main_process() and config.wandb.enable:
        try:
            run = setup_wandb(config)
            logger.info("Wandb is working!!!")
        except Exception as e:
            logger.warn("Wandb is not working!!!")
            print(e)

    is_pretrain = config.mode == "pt"

    logger.info(f"train_file: {config.train_file}")

    setup_seed(config.seed + get_rank())
    device = torch.device(config.device)

    train_loaders, test_name2loaders, train_media_types = setup_dataloaders(
        config, mode=config.mode
    )
    num_steps_per_epoch = sum(len(d) for d in train_loaders)
    config.scheduler.num_training_steps = num_steps_per_epoch * config.scheduler.epochs
    config.scheduler.num_warmup_steps = num_steps_per_epoch * config.scheduler.warmup_epochs
    # set cudnn.benchmark=True only when input size is fixed
    # https://discuss.pytorch.org/t/what-does-torch-backends-cudnn-benchmark-do/5936/3
    cudnn.benchmark = len(train_media_types) == 1

    print(f"\033[31m CURRENT NODE NAME: {os.environ['SLURMD_NODENAME']} dataloader is OK {datetime.datetime.now().strftime('%Y-%m-%d-%H_%M_%S')}!!! \033[0m")

    find_unused_parameters = config.model.get('find_unused_parameters', False)
    logger.info(f"find_unused_parameters={find_unused_parameters}")

    model_cls = eval(config.model.get('model_cls'))
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
        add_decoder=False,
        pretrain=is_pretrain,
        find_unused_parameters=find_unused_parameters,
    )

    if is_main_process() and config.wandb.enable:
        try:
            wandb.watch(model)
        except Exception as e:
            logger.warn("Wandb is not working!!!")
            print(e)

    best = 0
    best_epoch = 0
    if type(config.best_key) is str:
        best_key = [config.best_key, "t2v_r1"]
    elif type(config.best_key) is list and len(config.best_key) == 2:
        best_key = config.best_key
    else:
        raise NotImplementedError(config.best_key)

    best_ckpt_id = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    logger.info(f"Start training, start_epoch={start_epoch}")
    start_time = time.time()
    start_step = start_epoch * num_steps_per_epoch
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
                skip_num = global_step - start_step
            )

            if hasattr(config, "deepspeed") and config.deepspeed.enable:
                if config.get("save_latest", False):
                    tag = "ckpt_latest"
                else:
                    tag = f"ckpt_{epoch:02d}"

                client_state = {'epoch': epoch, 'global_step': global_step}
                model.save_checkpoint(config.output_dir, tag=tag, save_latest=False, client_state=client_state)

                logger.info(f"save ckpt file to local ({config.output_dir}/{tag})!!!") 
                if is_main_process() and config.get("delete_ds_optim_states", False):
                    if config.get("save_latest", False):
                        if epoch == (config.scheduler.epochs - 1): # last epoch
                            last_tag = "ckpt_latest"
                            last_ckpt_path = f"{config.output_dir}/{last_tag}"
                            if os.path.exists(last_ckpt_path):
                                logger.info(f"remove optim states in ({config.output_dir}/{last_tag})!!!")
                                for file in os.listdir(last_ckpt_path):
                                    if file.endswith('optim_states.pt'):
                                        os.remove(os.path.join(last_ckpt_path, file))
                    else:
                        last_tag = f"ckpt_{epoch-1:02d}"
                        last_ckpt_path = f"{config.output_dir}/{last_tag}"
                        if os.path.exists(last_ckpt_path):
                            logger.info(f"remove optim states in ({config.output_dir}/{last_tag})!!!")
                            for file in os.listdir(last_ckpt_path):
                                if file.endswith('optim_states.pt'):
                                    os.remove(os.path.join(last_ckpt_path, file))
                        
                        if epoch == (config.scheduler.epochs - 1): # last epoch
                            last_tag = f"ckpt_{epoch:02d}"
                            last_ckpt_path = f"{config.output_dir}/{last_tag}"
                            if os.path.exists(last_ckpt_path):
                                logger.info(f"remove optim states in ({config.output_dir}/{last_tag})!!!")
                                for file in os.listdir(last_ckpt_path):
                                    if file.endswith('optim_states.pt'):
                                        os.remove(os.path.join(last_ckpt_path, file))

            if is_main_process():
                if not (hasattr(config, "deepspeed") and config.deepspeed.enable):
                    state_dict = model_without_ddp.state_dict()
                    for k in config.get("no_save_params_prefix", []):
                        kk = [x for x in state_dict.keys() if x.startswith(k)]
                        logger.info(f"Not saving {len(kk)} params with prefix {k}")
                        for kkk in kk:
                            state_dict.pop(kkk)

                    save_obj = {
                        "model": state_dict,
                        "optimizer": optimizer.state_dict(),
                        "scheduler": scheduler.state_dict(),
                        "scaler": scaler.state_dict(),
                        "config": config,
                        "epoch": epoch,
                        "global_step": global_step,
                    }
                    try:
                        with io.BytesIO() as buffer:
                            torch.save(save_obj, buffer)
                            if config.get("save_latest", False):
                                client_ckpt.put(f"{ceph_ckpt_path}/ckpt_latest.pth", buffer.getvalue())
                                logger.info(f"Save to ceph ({ceph_ckpt_path}/ckpt_latest.pth)!!!")
                            else:
                                client_ckpt.put(f"{ceph_ckpt_path}/ckpt_{epoch:02d}.pth", buffer.getvalue())
                                logger.info(f"Save to ceph ({ceph_ckpt_path}/ckpt_{epoch:02d}.pth)!!!")
                    except Exception as e:
                        print(e)
                        if config.get("save_latest", False):
                            torch.save(save_obj, join(config.output_dir, "ckpt_latest.pth"))
                            logger.warn(f"Ceph is not working, save to local ({join(config.output_dir, 'ckpt_latest.pth')})!!!")
                        else:
                            torch.save(save_obj, join(config.output_dir, f"ckpt_{epoch:02d}.pth"))
                            logger.warn(f"Ceph is not working, save to local ({join(config.output_dir, f'ckpt_{epoch:02d}.pth')})!!!")



        if config.get("jump_evaluate", False) and not config.evaluate:
            logger.warn(f"Jump the evaluation'))!!!")
        else:
            try:
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
                    # log to wandb
                    if config.wandb.enable:
                        try:
                            for p, v in eval_res.items():
                                log_dict_to_wandb(v, step=global_step, prefix=p)
                        except Exception as e:
                            logger.warn("Wandb is not working!!!")
                            print(e)

                    try:
                        cur_recall = eval_res[best_key[0]][best_key[1]]
                    except Exception as e:
                        logger.warn(e)
                        print(e)
                        # print(eval_res)
                        cur_recall = best - 1

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

                    if not config.evaluate and cur_recall > best:
                        if not (hasattr(config, "deepspeed") and config.deepspeed.enable):
                            try:
                                with io.BytesIO() as buffer:
                                    torch.save(save_obj, buffer)
                                    client_ckpt.put(f"{ceph_ckpt_path}/ckpt_best_{best_ckpt_id}.pth", buffer.getvalue())
                                    logger.info(f"Save to ceph ({f'{ceph_ckpt_path}/ckpt_best_{best_ckpt_id}.pth'})!!!")
                            except Exception as e:
                                print(e)
                                torch.save(save_obj, join(config.output_dir, f"ckpt_best_{best_ckpt_id}.pth"))
                                logger.warn(f"Ceph is not working, save to local ({join(config.output_dir, f'ckpt_best_{best_ckpt_id}.pth')})!!!")
                        else:
                            
                            now_ckpt_path = f"{config.output_dir}/{tag}/mp_rank_00_model_states.pt"
                            best_ckpt_path = f"{config.output_dir}/best_mp_rank_00_model_states.pt"

                            if os.path.exists(now_ckpt_path):
                                shutil.copy(now_ckpt_path, best_ckpt_path)
                                logger.info(f"Copy {now_ckpt_path} to {best_ckpt_path}!!!")
                            else:
                                logger.warn(f"Can't find {now_ckpt_path}, there's some wrong!!!")

                        eval_file = "eval_res_best.json"
                        eval_res.to_json(join(config.output_dir, eval_file))
                        best = cur_recall
                        best_epoch = epoch
            except Exception as e:
                logger.warn("Something wrong when eval or save!!!")
                print(e)
                if config.evaluate:
                    raise e


        if config.evaluate:
            break
        
        start_step = global_step

        dist.barrier()

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    logger.info(f"Training time {total_time_str}")
    logger.info(f"best epoch {best_epoch} [best_key {best_key}]")
    logger.info(f"Checkpoints and Logs saved at {config.output_dir}")

    if is_main_process() and config.wandb.enable:
        try:
            run.finish()
        except Exception as e:
            logger.warn("Wandb is not working!!!")
            print(e)


if __name__ == "__main__":
    print(f"\033[31m NODE LIST: {os.environ['SLURM_NODELIST']} \033[0m")
    logger.info(f"NODE LIST: {os.environ['SLURM_NODELIST']}")
    cfg = setup_main()
    local_broadcast_process_authkey()
    main(cfg)
