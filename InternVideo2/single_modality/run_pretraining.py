import argparse
import datetime
import numpy as np
import time
import torch
import torch.backends.cudnn as cudnn
import json
import os
from functools import partial

from pathlib import Path
from timm.models import create_model
from optim_factory import (
    create_optimizer,
    get_parameter_groups,
)
from datasets import build_multi_pretraining_dataset
from engines.engine_for_pretraining import train_one_epoch
from utils import NativeScalerWithGradNormCount as NativeScaler
from utils import multiple_pretrain_samples_collate
import utils
from models import *


def get_args():
    parser = argparse.ArgumentParser('VideoMAE pre-training script', add_help=False)
    parser.add_argument('--batch_size', default=64, type=int)
    parser.add_argument('--epochs', default=800, type=int)
    parser.add_argument('--update_freq', default=1, type=int)
    parser.add_argument('--save_ckpt_freq', default=50, type=int)
    parser.add_argument('--steps_per_print', default=1, type=int)
    parser.add_argument('--use_ceph_checkpoint', action='store_true',
                        help="whether use ceph to save and load checkpoint, may be some bug now")
    parser.set_defaults(use_ceph_checkpoint=False)
    parser.add_argument('--ceph_checkpoint_prefix', default='', type=str,
                        help='prefix for checkpoint in ceph')
    parser.add_argument('--ckpt_path_split', default='/exp/', type=str,
                        help='string for splitting the ckpt_path')

    # Model parameters
    parser.add_argument('--model', default='pretrain_videomae_base_patch16_224', type=str, metavar='MODEL',
                        help='Name of model to train')
    parser.add_argument('--decoder_depth', default=4, type=int,
                        help='depth of decoder')
    parser.add_argument('--mask_type', default='tube', choices=['random', 'tube', 'attention'],
                        type=str, help='masked strategy of video tokens/patches')
    parser.add_argument('--mask_ratio', default=0.75, type=float,
                        help='ratio of the visual tokens/patches need be masked')
    parser.add_argument('--input_size', default=224, type=int,
                        help='videos input size for backbone')
    parser.add_argument('--drop_path', type=float, default=0.0, metavar='PCT',
                        help='Drop path rate (default: 0.0)')
    parser.add_argument('--normlize_target', default=True, type=bool,
                        help='normalized the target patch pixels')
    parser.add_argument('--tubelet_size', default=1, type=int,
                        help='temporal tube size for the patch embedding')
    parser.add_argument('--layer_scale_init_value', default=1e-5, type=float, 
                        help="0.1 for base, 1e-5 for large. set 0 to disable LayerScale")
    parser.add_argument('--layerscale_no_force_fp32', action='store_true',
                        help="Not force fp32 for LayerScale")
    parser.set_defaults(layerscale_no_force_fp32=False)
    parser.add_argument('--sep_pos_embed', action='store_true',
                        help="whether use seperable position embedding")
    parser.set_defaults(sep_pos_embed=False)

    # CLIP decpder parameters
    parser.add_argument('--clip_teacher', default='internvl_clip_6b', type=str,
                        help='Name of CLIP teacher')
    parser.add_argument('--clip_input_resolution', default=224, type=int,
                        help='input resolution of CLIP decoder')
    parser.add_argument('--clip_teacher_embed_dim', default=3200, type=int,
                        help='output dimension of CLIP decoder in the intermediate layers')
    parser.add_argument('--clip_teacher_final_dim', default=768, type=int,
                        help='output dimension of CLIP decoder in the final layer, 0 means w/o alignment')
    parser.add_argument('--clip_loss_ratio', default=[1, 1], type=float, nargs='+', metavar='BETA',
                        help='Loss ratio for middle features and final features (default: [1, 0.5])')
    parser.add_argument('--clip_norm_type', default='l2', type=str,
                        help='type of feature normalization')
    parser.add_argument('--clip_return_attn', action='store_true',
                        help="whether return CLIP attention")
    parser.set_defaults(clip_return_attn=False)
    parser.add_argument('--clip_return_layer', default=1, type=int,
                        help='number of CLIP return layers')
    parser.add_argument('--clip_teacher_return_interval', default=1, type=float,
                        help='interval of CLIP teacher return layers')
    parser.add_argument('--clip_student_return_interval', default=1, type=float,
                        help='interval of CLIP student return layers')
    
    # MAE decoder parameters
    parser.add_argument('--mae_teacher', default='clip_b16', type=str,
                        help='Name of MAE teacher')
    parser.add_argument('--mae_input_resolution', default=224, type=int,
                        help='input resolution of MAE decoder')
    parser.add_argument('--mae_tubelet_size', default=2, type=int,
                        help='tubelet size of MAE decoder')
    parser.add_argument('--mae_teacher_embed_dim', default=1408, type=int,
                        help='output dimension of MAE decoder')
    parser.add_argument('--mae_norm_type', default='l2', type=str,
                        help='type of feature normalization')
    parser.add_argument('--mae_loss_ratio', default=1., type=float,
                        help='ratio for MAE loss')
    parser.add_argument('--mae_return_layer', default=1, type=int,
                        help='number of MAE return layers')
    parser.add_argument('--mae_teacher_return_interval', default=1, type=float,
                        help='interval of MAE teacher return layers')
    parser.add_argument('--mae_student_return_interval', default=1, type=float,
                        help='interval of MAE student return layers')
    
    # Optimizer parameters
    parser.add_argument('--opt', default='adamw', type=str, metavar='OPTIMIZER',
                        help='Optimizer (default: "adamw"')
    parser.add_argument('--opt_eps', default=1e-6, type=float, metavar='EPSILON',
                        help='Optimizer Epsilon (default: 1e-6)')
    parser.add_argument('--opt_betas', default=[0.9, 0.98], type=float, nargs='+', metavar='BETA',
                        help='Optimizer Betas (default: [0.9, 0.98])')
    parser.add_argument('--clip_grad', type=float, default=3.0, metavar='NORM',
                        help='Clip gradient norm (default: 3.0)')
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                        help='SGD momentum (default: 0.9)')
    parser.add_argument('--weight_decay', type=float, default=0.05,
                        help='weight decay (default: 0.05)')
    parser.add_argument('--weight_decay_end', type=float, default=None, help="""Final value of the
        weight decay. We use a cosine schedule for WD. 
        (Set the same value with args.weight_decay to keep weight decay no change)""")

    parser.add_argument('--lr', type=float, default=1.5e-4, metavar='LR',
                        help='learning rate (default: 1.5e-4)')
    parser.add_argument('--warmup_lr', type=float, default=1e-6, metavar='LR',
                        help='warmup learning rate (default: 1e-6)')
    parser.add_argument('--min_lr', type=float, default=1e-5, metavar='LR',
                        help='lower lr bound for cyclic schedulers that hit 0 (1e-5)')

    parser.add_argument('--warmup_epochs', type=int, default=40, metavar='N',
                        help='epochs to warmup LR, if scheduler supports')
    parser.add_argument('--warmup_steps', type=int, default=-1, metavar='N',
                        help='epochs to warmup LR, if scheduler supports')
    parser.add_argument('--use_checkpoint', action='store_true')
    parser.set_defaults(use_checkpoint=False)
    parser.add_argument('--checkpoint_num', type=int, default=0)

    # Augmentation parameters
    parser.add_argument('--num_sample', type=int, default=1, help='Repeated_aug (default: 1)')
    parser.add_argument('--color_jitter', type=float, default=0.0, metavar='PCT',
                        help='Color jitter factor (default: 0.0)')
    parser.add_argument('--train_interpolation', type=str, default='bicubic',
                        help='Training interpolation (random, bilinear, bicubic default: "bicubic")')
    parser.add_argument('--flip', default=False, action='store_true',
                        help='whether flip the video in pretraining')

    # Dataset parameters
    parser.add_argument('--prefix', default='', type=str, help='prefix for data')
    parser.add_argument('--split', default=' ', type=str, help='split for metadata')
    parser.add_argument('--data_path', default='you_data_path', type=str,
                        help='dataset path')
    parser.add_argument('--imagenet_default_mean_and_std', default=True, action='store_true')
    parser.add_argument('--use_decord', action='store_true',
                        help='whether use decord to load video, otherwise load image')
    parser.add_argument('--no_use_decord', action='store_false', dest='use_decord')
    parser.set_defaults(use_decord=True)
    parser.add_argument('--num_segments', type=int, default=1)
    parser.add_argument('--num_frames', type=int, default=16)
    parser.add_argument('--sampling_rate', type=int, default=4)
    parser.add_argument('--output_dir', default='',
                        help='path where to save, empty for no saving')
    parser.add_argument('--log_dir', default=None,
                        help='path where to tensorboard log')
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--resume', default='', help='resume from checkpoint')
    parser.add_argument('--auto_resume', action='store_true')
    parser.add_argument('--no_auto_resume', action='store_false', dest='auto_resume')
    parser.set_defaults(auto_resume=True)

    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='start epoch')
    parser.add_argument('--test_best', action='store_true',
                        help='Whether test the best model')
    parser.add_argument('--num_workers', default=10, type=int)
    parser.add_argument('--pin_mem', action='store_true',
                        help='Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.')
    parser.add_argument('--no_pin_mem', action='store_false', dest='pin_mem',
                        help='')
    parser.set_defaults(pin_mem=True)

    # distributed training parameters
    parser.add_argument('--world_size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--local_rank', default=-1, type=int)
    parser.add_argument('--dist_on_itp', action='store_true')
    parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')
    
    parser.add_argument('--enable_deepspeed',
                        action='store_true', default=False)
    parser.add_argument('--bf16', default=False, action='store_true')
    parser.add_argument('--zero_stage', default=0, type=int,
                        help='ZeRO optimizer stage (default: 0)')
    
    known_args, _ = parser.parse_known_args()

    if known_args.enable_deepspeed:
        try:
            import deepspeed
            parser = deepspeed.add_config_arguments(parser)
            ds_init = deepspeed.initialize
        except:
            print("Please install DeepSpeed")
            exit(0)
    else:
        ds_init = None

    return parser.parse_args(), ds_init


def get_model(args):
    print(f"Creating model: {args.model}")
    model = create_model(
        args.model,
        pretrained=False,
        drop_path_rate=args.drop_path,
        num_frames=args.num_frames//(args.mae_tubelet_size//args.tubelet_size),
        tubelet_size=args.tubelet_size,
        sep_pos_embed=args.sep_pos_embed,
        use_checkpoint=args.use_checkpoint,
        checkpoint_num=args.checkpoint_num,
        init_values=args.layer_scale_init_value,
        layerscale_no_force_fp32=args.layerscale_no_force_fp32,
        clip_teacher_embed_dim=args.clip_teacher_embed_dim,
        clip_teacher_final_dim=args.clip_teacher_final_dim,
        clip_norm_type=args.clip_norm_type,
        clip_return_layer=args.clip_return_layer,
        clip_student_return_interval=args.clip_student_return_interval,
        mae_teacher_embed_dim=args.mae_teacher_embed_dim,
        mae_norm_type=args.mae_norm_type,
        mae_return_layer=args.mae_return_layer,
        mae_student_return_interval=args.mae_student_return_interval,
    )
    return model


def main(args, ds_init):
    utils.init_distributed_mode(args)

    if ds_init is not None:
        utils.create_internvideo2_ds_config(args)

    print(args)

    device = torch.device(args.device)

    # fix the seed for reproducibility
    seed = args.seed + utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)

    cudnn.benchmark = True

    model = get_model(args)
    patch_size = model.patch_embed.patch_size
    print("Patch size = %s" % str(patch_size))
    print("Tubelet size = %s" % str(args.tubelet_size))
    args.window_size = (args.num_frames // args.tubelet_size, args.input_size // patch_size[0], args.input_size // patch_size[1])
    args.patch_size = patch_size

    # CLIP teacher model
    print(f'CLIP Teacher model: {args.clip_teacher}')
    clip_teacher_model = eval(args.clip_teacher)(
        img_size=args.clip_input_resolution,
        clip_norm_type=args.clip_norm_type,
        return_attn=args.clip_return_attn,
        clip_return_layer=args.clip_return_layer,
        clip_return_interval=args.clip_teacher_return_interval
    )

    # MAE teacher model
    print(f'MAE Teacher model: {args.mae_teacher}')
    mae_teacher_model = eval(args.mae_teacher)(
        img_size=args.mae_input_resolution,
        tubelet_size=args.mae_tubelet_size,
        mae_norm_type=args.mae_norm_type,
        mae_return_layer=args.mae_return_layer,
        mae_return_interval=args.mae_teacher_return_interval
    )

    # get dataset
    dataset_train = build_multi_pretraining_dataset(args)

    num_tasks = utils.get_world_size()
    global_rank = utils.get_rank()
    sampler_rank = global_rank
    num_training_steps_per_epoch = len(dataset_train) // args.batch_size // num_tasks

    sampler_train = torch.utils.data.DistributedSampler(dataset_train, num_replicas=num_tasks, rank=sampler_rank, shuffle=True)
    print("Sampler_train = %s" % str(sampler_train))

    if global_rank == 0 and args.log_dir is not None:
        os.makedirs(args.log_dir, exist_ok=True)
        log_writer = utils.TensorboardLogger(log_dir=args.log_dir)
    else:
        log_writer = None

    if args.num_sample > 1:
        collate_func = partial(multiple_pretrain_samples_collate, fold=False)
    else:
        collate_func = None

    data_loader_train = torch.utils.data.DataLoader(
        dataset_train, sampler=sampler_train,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=True,
        collate_fn=collate_func,
        worker_init_fn=utils.seed_worker,
        persistent_workers=True
    )

    model.to(device)
    clip_teacher_model.to(device)
    mae_teacher_model.to(device)
    model_without_ddp = model
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print("Model = %s" % str(model_without_ddp))
    print('number of params: {} M'.format(n_parameters / 1e6))

    total_batch_size = args.batch_size * utils.get_world_size()

    args.lr = args.lr * total_batch_size * args.num_sample / 256
    args.min_lr = args.min_lr * total_batch_size * args.num_sample / 256
    args.warmup_lr = args.warmup_lr * total_batch_size * args.num_sample / 256
    print("LR = %.8f" % args.lr)
    print("Batch size = %d" % total_batch_size)
    print("Repeated sample = %d" % args.num_sample)
    print("Number of training steps = %d" % num_training_steps_per_epoch)
    print("Number of training examples per epoch = %d" % (total_batch_size * num_training_steps_per_epoch))

    skip_weight_decay_list = model.no_weight_decay()
    print("Skip weight decay list: ", skip_weight_decay_list)

    if args.enable_deepspeed:
        loss_scaler = None
        optimizer_params = get_parameter_groups(
            model, args.weight_decay, skip_weight_decay_list
        )
        model, optimizer, _, _ = ds_init(
            args=args, model=model, model_parameters=optimizer_params,
            dist_init_required=not args.distributed,
        )

        print("model.gradient_accumulation_steps() = %d" %
              model.gradient_accumulation_steps())
        assert model.gradient_accumulation_steps() == args.update_freq
    else:
        if args.distributed:
            model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu], find_unused_parameters=False)
            model_without_ddp = model.module

        optimizer = create_optimizer(args, model_without_ddp)
        loss_scaler = NativeScaler()

    print("Use step level LR & WD scheduler!")
    lr_schedule_values = utils.cosine_scheduler(
        args.lr, args.min_lr, args.epochs, num_training_steps_per_epoch,
        warmup_epochs=args.warmup_epochs, warmup_steps=args.warmup_steps,
    )
    if args.weight_decay_end is None:
        args.weight_decay_end = args.weight_decay
    wd_schedule_values = utils.cosine_scheduler(args.weight_decay, args.weight_decay_end, args.epochs, num_training_steps_per_epoch)
    print("Max WD = %.7f, Min WD = %.7f" % (max(wd_schedule_values), min(wd_schedule_values)))

    ceph_args = {
        'use_ceph_checkpoint': args.use_ceph_checkpoint,
        'ceph_checkpoint_prefix': args.ceph_checkpoint_prefix,
        'ckpt_path_split': args.ckpt_path_split,
        'local_rank': args.gpu,
    }
    if ceph_args['use_ceph_checkpoint']:
        print("Will automatically upload model on ceph")
        assert ceph_args['ceph_checkpoint_prefix'] != '', "Should set prefix for ceph checkpoint!"

    utils.auto_load_model(
        args=args, model=model, model_without_ddp=model_without_ddp, optimizer=optimizer, loss_scaler=loss_scaler,
        ceph_args=ceph_args,
    )
    torch.cuda.empty_cache()
    print(f"Start training for {args.epochs} epochs")
    print(f"Use bf16 {args.bf16}")
    print(f"Mask ratio: {args.mask_ratio}")
    print(f"Mask typr: {args.mask_type}")
    distill_final_features = args.clip_teacher_final_dim > 0
    print(f"Distill final (AttnPoll) features of teacher: {distill_final_features}")
    print(f"Loss ratio: {args.clip_loss_ratio}")
    
    start_time = time.time()
    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            data_loader_train.sampler.set_epoch(epoch)
        if log_writer is not None:
            log_writer.set_step(epoch * num_training_steps_per_epoch)
        train_stats = train_one_epoch(
            model, data_loader_train,
            optimizer, device, epoch, loss_scaler,
            args.clip_grad, log_writer=log_writer,
            start_steps=epoch * num_training_steps_per_epoch,
            lr_schedule_values=lr_schedule_values,
            wd_schedule_values=wd_schedule_values,
            clip_teacher_model=clip_teacher_model, 
            clip_input_resolution=args.clip_input_resolution,
            distill_final_features=distill_final_features,
            clip_loss_ratio=args.clip_loss_ratio,
            mae_teacher_model=mae_teacher_model, 
            mae_input_resolution=args.mae_input_resolution,
            mae_loss_ratio=args.mae_loss_ratio,
            td_ratio=args.mae_tubelet_size//args.tubelet_size,
            mask_type=args.mask_type,
            mask_ratio=args.mask_ratio,
            bf16=args.bf16,
        )
        if args.output_dir:
            if (epoch + 1) % args.save_ckpt_freq == 0 or epoch + 1 == args.epochs:
                utils.save_model(
                    args=args, model=model, model_without_ddp=model_without_ddp, 
                    optimizer=optimizer, loss_scaler=loss_scaler, epoch=epoch,
                    ceph_args=ceph_args,
                )
            utils.save_model(
                args=args, model=model, model_without_ddp=model_without_ddp, 
                optimizer=optimizer, loss_scaler=loss_scaler, epoch=epoch,
                model_name='latest', ceph_args=ceph_args,
            )

        log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                     'epoch': epoch, 'n_parameters': n_parameters}

        if args.output_dir and utils.is_main_process():
            if log_writer is not None:
                log_writer.flush()
            with open(os.path.join(args.output_dir, "log.txt"), mode="a", encoding="utf-8") as f:
                f.write(json.dumps(log_stats) + "\n")

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))


if __name__ == '__main__':
    opts, ds_init = get_args()
    if opts.output_dir:
        Path(opts.output_dir).mkdir(parents=True, exist_ok=True)
    main(opts, ds_init)
