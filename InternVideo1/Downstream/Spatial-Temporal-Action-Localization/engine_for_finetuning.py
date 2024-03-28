from itertools import count
import os
import numpy as np
import math
import sys
import time
import datetime
import logging
from typing import Iterable, Optional
import torch
import torch.nn as nn

import torch.nn.functional as F

from timm.data import Mixup
from timm.utils import accuracy, ModelEma
import utils
from alphaction.modeling.utils import cat
from alphaction.structures.bounding_box import BoxList
from data.ava_eval import do_ava_evaluation
import pdb


def train_class_batch(model, samples, boxes):
    outputs = model(samples, boxes)
    labels = cat([proposal.get_field("labels") for proposal in boxes], dim=0)  # [n,80]
    assert outputs.shape[1] == labels.shape[1], \
        "The shape of tensor class logits doesn't match the label tensor."
    # loss = criterion(outputs, target)
    batch_size = outputs.shape[0]
    loss = F.binary_cross_entropy_with_logits(outputs, labels.to(dtype=torch.float32), reduction='mean')
    loss = loss * batch_size

    return loss, outputs


def get_loss_scale_for_deepspeed(model):
    optimizer = model.optimizer
    return optimizer.loss_scale if hasattr(optimizer, "loss_scale") else optimizer.cur_scale


def train_one_epoch(model: torch.nn.Module,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, loss_scaler, max_norm: float = 0,
                    model_ema: Optional[ModelEma] = None, mixup_fn: Optional[Mixup] = None, log_writer=None,
                    start_steps=None, lr_schedule_values=None, wd_schedule_values=None,
                    num_training_steps_per_epoch=None, update_freq=None):
    model.train(True)
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    metric_logger.add_meter('min_lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 10

    if loss_scaler is None:
        model.zero_grad()
        model.micro_steps = 0
    else:
        optimizer.zero_grad()

    for data_iter_step, (samples, boxes, _) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
        step = data_iter_step // update_freq
        if step >= num_training_steps_per_epoch:
            continue
        it = start_steps + step  # global training iteration
        # Update LR & WD for the first acc
        if lr_schedule_values is not None or wd_schedule_values is not None and data_iter_step % update_freq == 0:
            for i, param_group in enumerate(optimizer.param_groups):
                if lr_schedule_values is not None:
                    param_group["lr"] = lr_schedule_values[it] * param_group["lr_scale"]
                if wd_schedule_values is not None and param_group["weight_decay"] > 0:
                    param_group["weight_decay"] = wd_schedule_values[it]

        samples = samples.to(device, non_blocking=True)
        boxes = [box.to(device=device) for box in boxes]  # boxlist
        # targets = targets.to(device, non_blocking=True)

        # if mixup_fn is not None:
        #     samples, targets = mixup_fn(samples, targets)

        if loss_scaler is None:
            samples = samples.half()
            loss, _ = train_class_batch(
                model, samples, boxes)
        else:
            with torch.cuda.amp.autocast():
                loss, _ = train_class_batch(
                    model, samples, boxes)

        loss_value = loss.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            sys.exit(1)

        if loss_scaler is None:
            loss /= update_freq
            model.backward(loss)
            model.step()

            if (data_iter_step + 1) % update_freq == 0:
                # model.zero_grad()
                # Deepspeed will call step() & model.zero_grad() automatic
                if model_ema is not None:
                    model_ema.update(model)
            grad_norm = None
            loss_scale_value = get_loss_scale_for_deepspeed(model)
        else:
            # this attribute is added by timm on one optimizer (adahessian)
            is_second_order = hasattr(optimizer, 'is_second_order') and optimizer.is_second_order
            loss /= update_freq
            grad_norm = loss_scaler(loss, optimizer, clip_grad=max_norm,
                                    parameters=model.parameters(), create_graph=is_second_order,
                                    update_grad=(data_iter_step + 1) % update_freq == 0)
            if (data_iter_step + 1) % update_freq == 0:
                optimizer.zero_grad()
                if model_ema is not None:
                    model_ema.update(model)
            loss_scale_value = loss_scaler.state_dict()["scale"]

        torch.cuda.synchronize()

        # if mixup_fn is None:
        #     class_acc = (output.max(-1)[-1] == targets).float().mean()
        # else:
        #     class_acc = None
        metric_logger.update(loss=loss_value)
        # metric_logger.update(class_acc=class_acc)
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
            # log_writer.update(class_acc=class_acc, head="loss")
            log_writer.update(loss_scale=loss_scale_value, head="opt")
            log_writer.update(lr=max_lr, head="opt")
            log_writer.update(min_lr=min_lr, head="opt")
            log_writer.update(weight_decay=weight_decay_value, head="opt")
            log_writer.update(grad_norm=grad_norm, head="opt")

            log_writer.set_step()

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


class PostProcessor(nn.Module):

    def forward(self, class_logits, boxes):
        # boxes should be (#detections,4)
        # prob should be calculated in different way.
        class_logits = torch.sigmoid(class_logits)  # [n,80]
        # 给action_prob乘以box分数
        box_scores = cat([box.get_field("scores") for box in boxes], dim=0)
        box_scores = box_scores.reshape(class_logits.shape[0], 1)  # [B,1]
        action_prob = class_logits * box_scores

        image_shapes = [box.size for box in boxes]
        boxes_per_image = [len(box) for box in boxes]
        box_tensors = [a.bbox for a in boxes]

        action_prob = action_prob.split(boxes_per_image, dim=0)  # [rois,80]->[bs,per_roi,80]

        results = []
        for prob, boxes_per_image, image_shape in zip(
                action_prob, box_tensors, image_shapes
        ):
            boxlist = self.prepare_boxlist(boxes_per_image, prob, image_shape)
            results.append(boxlist)
        return results

    def prepare_boxlist(self, boxes, scores, image_shape):
        boxlist = BoxList(boxes, image_shape, mode="xyxy")
        boxlist.add_field("scores", scores)
        return boxlist


@torch.no_grad()
def validation_one_epoch(data_loader, model, device, output_dir, epoch, log_writer=None):
    if not utils.is_main_process():
        return

    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Val:'

    # switch to evaluation mode
    model.eval()

    logging.info("Start evaluation on ava_v2.2 dataset({} videos).".format(data_loader.num_samples))
    start_time = time.time()

    cpu_device = torch.device("cpu")
    results_dict = {}
    postprocess = PostProcessor()
    for batch in metric_logger.log_every(data_loader, 10, header):
        # pdb.set_trace()
        videos = batch[0]
        boxes = batch[1]
        video_ids = batch[2]

        videos = videos.to(device, non_blocking=True)
        boxes = [box.to(device=device) for box in boxes]  # boxlist
        # target = target.to(device, non_blocking=True)

        # compute output
        with torch.cuda.amp.autocast():
            output = model(videos, boxes)  # [n,80]
            output = postprocess(output, boxes)
            output = [o.to(cpu_device) for o in output]
            results_dict.update(
                {video_id: result for video_id, result in zip(video_ids, output)}
            )
    # pdb.set_trace()
    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=total_time))
    logging.info(
        "Total inference time: {}".format(total_time_str)
    )

    # convert a dict where the key is the index in a list
    video_ids = list(sorted(results_dict.keys()))
    if len(video_ids) != video_ids[-1] + 1:
        logging.warning(
            "Number of videos that were gathered from multiple processes is not "
            "a contiguous set. Some images might be missing from the evaluation"
        )
    # convert to a list
    predictions = [results_dict[i] for i in video_ids]

    logging.info("Performing ava evaluation")

    output_folder = os.path.join(output_dir, "inference")
    os.makedirs(output_folder, exist_ok=True)

    eval_res = do_ava_evaluation(
        dataset=data_loader.dataset,
        predictions=predictions,
        output_folder=output_folder,
    )

    if log_writer is not None:
        eval_res, _ = eval_res
        total_mAP = eval_res['PascalBoxes_Precision/mAP@0.5IOU']
        log_writer.update(map=total_mAP, head="perf", step=epoch)

# 以下没用到
@torch.no_grad()
def final_test(data_loader, model, device, file):
    criterion = torch.nn.CrossEntropyLoss()

    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Test:'

    # switch to evaluation mode
    model.eval()
    final_result = []
    
    for batch in metric_logger.log_every(data_loader, 10, header):
        videos = batch[0]
        target = batch[1]
        ids = batch[2]
        chunk_nb = batch[3]
        split_nb = batch[4]
        videos = videos.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)

        # compute output
        with torch.cuda.amp.autocast():
            output = model(videos)
            loss = criterion(output, target)

        for i in range(output.size(0)):
            string = "{} {} {} {} {}\n".format(ids[i], \
                                                str(output.data[i].cpu().numpy().tolist()), \
                                                str(int(target[i].cpu().numpy())), \
                                                str(int(chunk_nb[i].cpu().numpy())), \
                                                str(int(split_nb[i].cpu().numpy())))
            final_result.append(string)

        acc1, acc5 = accuracy(output, target, topk=(1, 5))

        batch_size = videos.shape[0]
        metric_logger.update(loss=loss.item())
        metric_logger.meters['acc1'].update(acc1.item(), n=batch_size)
        metric_logger.meters['acc5'].update(acc5.item(), n=batch_size)

    if not os.path.exists(file):
        os.mknod(file)
    with open(file, 'w') as f:
        f.write("{}, {}\n".format(acc1, acc5))
        for line in final_result:
            f.write(line)
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print('* Acc@1 {top1.global_avg:.3f} Acc@5 {top5.global_avg:.3f} loss {losses.global_avg:.3f}'
          .format(top1=metric_logger.acc1, top5=metric_logger.acc5, losses=metric_logger.loss))

    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


def merge(eval_path, num_tasks):
    dict_feats = {}
    dict_label = {}
    dict_pos = {}
    print("Reading individual output files")

    for x in range(num_tasks):
        file = os.path.join(eval_path, str(x) + '.txt')
        lines = open(file, 'r').readlines()[1:]
        for line in lines:
            line = line.strip()
            name = line.split('[')[0]
            label = line.split(']')[1].split(' ')[1]
            chunk_nb = line.split(']')[1].split(' ')[2]
            split_nb = line.split(']')[1].split(' ')[3]
            data = np.fromstring(line.split('[')[1].split(']')[0], dtype=np.float, sep=',')
            if not name in dict_feats:
                dict_feats[name] = []
                dict_label[name] = 0
                dict_pos[name] = []
            if chunk_nb + split_nb in dict_pos[name]:
                continue
            dict_feats[name].append(data)
            dict_pos[name].append(chunk_nb + split_nb)
            dict_label[name] = label
    print("Computing final results")

    input_lst = []
    print(len(dict_feats))
    for i, item in enumerate(dict_feats):
        input_lst.append([i, item, dict_feats[item], dict_label[item]])
    from multiprocessing import Pool
    p = Pool(64)
    ans = p.map(compute_video, input_lst)
    top1 = [x[1] for x in ans]
    top5 = [x[2] for x in ans]
    pred = [x[0] for x in ans]
    label = [x[3] for x in ans]
    final_top1 ,final_top5 = np.mean(top1), np.mean(top5)
    return final_top1*100 ,final_top5*100

def compute_video(lst):
    i, video_id, data, label = lst
    feat = [x for x in data]
    feat = np.mean(feat, axis=0)
    pred = np.argmax(feat)
    top1 = (int(pred) == int(label)) * 1.0
    top5 = (int(label) in np.argsort(-feat)[:5]) * 1.0
    return [pred, top1, top5, int(label)]
