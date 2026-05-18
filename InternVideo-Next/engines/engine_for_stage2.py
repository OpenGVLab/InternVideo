import math
import time
import sys
from typing import Iterable
import torch
import torch.distributed as dist
import utils

from .jepa_mask import get_jepa_mask_collator


def get_loss_scale_for_deepspeed(model):
    optimizer = model.optimizer
    loss_scale = None
    if hasattr(optimizer, 'loss_scale'):
        loss_scale = optimizer.loss_scale
    elif hasattr(optimizer, 'cur_scale'):
        loss_scale = optimizer.cur_scale
    return loss_scale, optimizer._global_grad_norm


def train_one_epoch(
        model: torch.nn.Module, data_loader: Iterable, optimizer: torch.optim.Optimizer,
        device: torch.device, epoch: int, loss_scaler, max_norm: float = 0,
        log_writer=None, lr_scheduler=None, start_steps=None,
        lr_schedule_values=None, wd_schedule_values=None, 
        clip_teacher_model=None, clip_input_frame=8, clip_input_resolution=378,
        distill_final_features=True,
        clip_loss_ratio=[1, 1, 1],
        momentum=0.998,
        mask_type='attention', mask_ratio=0., reconstruction_ratio=0.5,
        bf16=False, diffusion_loss_only=False, umt_loss_only=False, wo_middle_loss=False,
    ):
    model.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    metric_logger.add_meter('min_lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 1

    if loss_scaler is None:
        model.zero_grad()
        model.micro_steps = 0
    else:
        optimizer.zero_grad()

    if bf16:
        datatype = torch.bfloat16
    else:
        datatype = torch.float16

    mask_generator = get_jepa_mask_collator(num_frames=clip_input_frame)

    for step, batch in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
        # assign learning rate & weight decay for each step
        it = start_steps + step  # global training iteration
        if lr_schedule_values is not None or wd_schedule_values is not None:
            for i, param_group in enumerate(optimizer.param_groups):
                if lr_schedule_values is not None:
                    param_group["lr"] = lr_schedule_values[it] * param_group["lr_scale"]
                if wd_schedule_values is not None and param_group["weight_decay"] > 0:
                    param_group["weight_decay"] = wd_schedule_values[it]

        videos, bool_masked_pos = batch
        videos = videos.to(device, non_blocking=True)

        if mask_type in ['attention']:
            bool_masked_pos = None
        else:
            bool_masked_pos = bool_masked_pos.flatten(1)
            bool_masked_pos = torch.cat((torch.zeros(bool_masked_pos.shape[0], 1), bool_masked_pos), dim=1)
            bool_masked_pos = bool_masked_pos.to(device, non_blocking=True).to(torch.bool)

        with torch.no_grad():
            B, C, T, H, W = videos.shape

            mask_v1 = mask_generator(batch_size = B).bool()
            mask_v2 = mask_generator(batch_size = B, random_index=1).bool()

            if H != clip_input_resolution:
                clip_videos = torch.nn.functional.interpolate(
                    videos.view(B, C*T, H, W), 
                    size=(clip_input_resolution, clip_input_resolution), 
                    mode='bicubic', align_corners=False
                )
                clip_videos = clip_videos.view(B, C, T, clip_input_resolution, clip_input_resolution)
            else:
                clip_videos = videos

            with torch.cuda.amp.autocast(dtype=datatype):
                norm_clip_middle, norm_clip_final = clip_teacher_model(clip_videos, embed_only=True)

        if loss_scaler is None:
            videos = videos.bfloat16() if bf16 else videos.half()
            with torch.cuda.amp.autocast(dtype=datatype):
                outputs_clip_middle_v1, _ = model(videos, mask=mask_v1)
                outputs_clip_middle_v2, _ = model(videos, mask=mask_v2)

            # K = norm_clip_middle.shape[0]
            C_CLIP = norm_clip_middle.shape[-1]

            # middle_loss_mask = mask.repeat(K, 1, 1)
            targets_clip_middle_vis_v1 = norm_clip_middle[mask_v1].reshape(B, -1, C_CLIP)
            targets_clip_middle_vis_v1 = targets_clip_middle_vis_v1 / targets_clip_middle_vis_v1.norm(dim=-1, keepdim=True)
            targets_clip_middle_vis_v2 = norm_clip_middle[mask_v2].reshape(B, -1, C_CLIP)
            targets_clip_middle_vis_v2 = targets_clip_middle_vis_v2 / targets_clip_middle_vis_v2.norm(dim=-1, keepdim=True)

            loss_clip_middle = (2 - 2 * (outputs_clip_middle_v1 * targets_clip_middle_vis_v1).sum(dim=-1)).mean()
            loss_clip_middle += (2 - 2 * (outputs_clip_middle_v2 * targets_clip_middle_vis_v2).sum(dim=-1)).mean()
        else:
            assert False

        loss = loss_clip_middle * clip_loss_ratio[0]

        loss_value = loss.item()

        loss_list = [torch.zeros_like(loss) for _ in range(dist.get_world_size())]
        dist.all_gather(loss_list, loss)
        loss_list = torch.tensor(loss_list)
        all_loss_mean_value = loss_list.mean().item()
        metric_logger.update(all_loss_mean=all_loss_mean_value)

        loss_list_isnan = torch.isnan(loss_list).any()
        loss_list_isinf = torch.isinf(loss_list).any()
        if loss_list_isnan or loss_list_isinf:
            print(f"Skipping iteration {step} due to NaN loss detected on one or more nodes")
            # Instead of continue, we'll zero out the gradients and skip the update
            if loss_scaler is None:
                model.zero_grad()
            else:
                optimizer.zero_grad()
            
            # Still need to perform the synchronization steps
            loss_value = float(0.)
            loss_clip_middle_value = float(0.)
            # loss_clip_final_value = float(0.)
            # diffusion_loss_value = float(0.)
            loss_scale_value = 0
            grad_norm = 0
        else:

            if loss_scaler is None:
                model.backward(loss)
                model.step()
                loss_scale_value, grad_norm = get_loss_scale_for_deepspeed(model)
            else:
                optimizer.zero_grad()
                is_second_order = hasattr(optimizer, 'is_second_order') and optimizer.is_second_order
                grad_norm = loss_scaler(loss, optimizer, clip_grad=max_norm,
                                    parameters=model.parameters(), create_graph=is_second_order)
                loss_scale_value = loss_scaler.state_dict()["scale"]

            loss_clip_middle_value = loss_clip_middle.item()
            # loss_clip_final_value = loss_clip_final.item()
            # diffusion_loss_value = diffusion_loss.item()

        torch.cuda.synchronize()
        
        with torch.no_grad():
            for param_q, param_k in zip(model.module.parameters(), clip_teacher_model.parameters()):
                param_k.data.mul_(momentum).add_((1.-momentum) * param_q.detach().data)

        # Update metrics regardless of NaN status
        metric_logger.update(loss=loss_value)
        metric_logger.update(loss_clip_middle=loss_clip_middle_value)
        # metric_logger.update(loss_clip_final=loss_clip_final_value)
        # metric_logger.update(diffusion_loss=diffusion_loss_value)
        metric_logger.update(loss_scale=loss_scale_value)
        
        min_lr = 10.
        max_lr = 0.
        for group in optimizer.param_groups:
            min_lr = min(min_lr, group["lr"])
            max_lr = max(max_lr, group["lr"])

        metric_logger.update(lr=max_lr)
        metric_logger.update(min_lr=min_lr)
        weight_decay_value = None
        for group in optimizer.param_groups:
            if group["weight_decay"] > 0:
                weight_decay_value = group["weight_decay"]
        metric_logger.update(weight_decay=weight_decay_value)
        metric_logger.update(grad_norm=grad_norm)

        if log_writer is not None:
            log_writer.update(loss=loss_value, head="loss")
            log_writer.update(loss_clip_middle=loss_clip_middle_value, head="loss")
            # log_writer.update(loss_clip_final=loss_clip_final_value, head="loss")
            # log_writer.update(diffusion_loss=diffusion_loss_value, head="loss")
            log_writer.update(loss_scale=loss_scale_value, head="opt")
            log_writer.update(lr=max_lr, head="opt")
            log_writer.update(min_lr=min_lr, head="opt")
            log_writer.update(weight_decay=weight_decay_value, head="opt")
            log_writer.update(grad_norm=grad_norm, head="opt")
            log_writer.set_step()

        if lr_scheduler is not None:
            lr_scheduler.step_update(start_steps + step)

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    timestep = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
    print(f"[{timestep}] Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}