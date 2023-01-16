import torch
import io
import random

from transformers import (
    get_polynomial_decay_schedule_with_warmup,
    get_cosine_schedule_with_warmup,
)
from CoTrain.modules.dist_utils import all_gather
from CoTrain.modules.objectives import compute_irtr_recall, compute_decouple_irtr_recall, compute_ind_irtr_recall
# from CoTrain.gadgets.my_metrics import Accuracy, VQAScore, Scalar
from CoTrain.gadgets.my_metrics import VQAScore
from torchmetrics import Accuracy
from torchmetrics import MeanMetric as Scalar
from CoTrain.datasets import client
from .clip_param_keys import clip_param_keys


def set_split_metrics(pl_module, split, loss_names):
    for k, v in loss_names.items():
        if v < 1:
            continue
        if k == "vqa":
            setattr(pl_module, f"{split}_vqa_score", VQAScore())
            setattr(pl_module, f"{split}_{k}_loss", Scalar())
        # vcr
        elif k == "vcr_q2a":
            setattr(pl_module, f"{split}_{k}_loss", Scalar())
            setattr(pl_module, f"{split}_{k}_accuracy", Accuracy())
            setattr(pl_module, f"{split}_vcr_qar_loss", Scalar())
            setattr(pl_module, f"{split}_vcr_qar_accuracy", Accuracy())
        elif k == "mc_vqa":
            setattr(pl_module, f"{split}_{k}_loss", Scalar())
            setattr(pl_module, f"{split}_{k}_accuracy", Accuracy())
        elif k == "openend_vqa":
            setattr(pl_module, f"{split}_vqa_score", VQAScore())
            setattr(pl_module, f"{split}_vqa_loss", Scalar())
            setattr(pl_module, f"{split}_{k}_loss", Scalar())
            setattr(pl_module, f"{split}_{k}_accuracy", Accuracy(ignore_index=-100))
        elif k == "vcop":
            setattr(pl_module, f"{split}_vcop_score", VQAScore())
            setattr(pl_module, f"{split}_vcop_loss", Scalar())
            setattr(pl_module, f"{split}_{k}_loss", Scalar())
            setattr(pl_module, f"{split}_{k}_accuracy", Accuracy())
        elif k == "multiple_choice":
            setattr(pl_module, f"{split}_{k}_accuracy", Accuracy())
            setattr(pl_module, f"{split}_{k}_loss", Scalar())
        elif k == "nlvr2":
            if split == "train":
                setattr(pl_module, f"train_{k}_accuracy", Accuracy())
                setattr(pl_module, f"train_{k}_loss", Scalar())
            else:
                setattr(pl_module, f"dev_{k}_accuracy", Accuracy())
                setattr(pl_module, f"dev_{k}_loss", Scalar())
                setattr(pl_module, f"test_{k}_accuracy", Accuracy())
                setattr(pl_module, f"test_{k}_loss", Scalar())
        elif k == "irtr":
            setattr(pl_module, f"{split}_irtr_loss", Scalar())
        elif k == "mppd" or k == "mpfr":
            setattr(pl_module, f"{split}_{k}_loss", Scalar())
        elif k == "vtm":
            setattr(pl_module, f"{split}_{k}_accuracy", Accuracy())
            setattr(pl_module, f"{split}_{k}_loss", Scalar())
            setattr(pl_module, f"{split}_{k}_dino_loss", Scalar())
            setattr(pl_module, f"{split}_{k}_wpa_loss", Scalar())
        # add for image text contrastive learning
        elif k == "itc":
            setattr(pl_module, f"{split}_{k}_loss", Scalar())
        elif k == "dino":
            setattr(pl_module, f"{split}_{k}_loss", Scalar())
        elif k == "contrastive":
            setattr(pl_module, f"{split}_{k}_loss", Scalar())
            setattr(pl_module, f"{split}_{k}_image_accuracy", Accuracy())
            setattr(pl_module, f"{split}_{k}_text_accuracy", Accuracy())
        elif k == "zs_classify":
            setattr(pl_module, f"{split}_{k}_accuracy", Accuracy())
        elif k == "mlm":
            setattr(pl_module, f"{split}_{k}_accuracy", Accuracy(ignore_index=-100))
            setattr(pl_module, f"{split}_{k}_loss", Scalar())
        else:
            setattr(pl_module, f"{split}_{k}_accuracy", Accuracy())
            setattr(pl_module, f"{split}_{k}_loss", Scalar())


def set_metrics(pl_module):
    set_split_metrics(pl_module, "train", pl_module.hparams.config["loss_names"])
    if len(pl_module.hparams.config["val_datasets"]) == 0:
        set_split_metrics(pl_module, "val", pl_module.hparams.config["loss_names"])
    else:
        set_split_metrics(pl_module, "val", pl_module.hparams.config["val_loss_names"])


def epoch_wrapup(pl_module):
    phase = "train" if pl_module.training else "val"
    the_metric = 0
    the_metric_qar = 0
    if pl_module.hparams.config["get_recall_metric"] and not pl_module.training:
        (ir_r1, ir_r5, ir_r10, tr_r1, tr_r5, tr_r10) = compute_irtr_recall(pl_module)
        if torch.distributed.get_rank() == 0:
            print((ir_r1, ir_r5, ir_r10, tr_r1, tr_r5, tr_r10), pl_module.global_step)
        pl_module.logger.experiment.add_scalar(
            "recalls/ir_r1", ir_r1, pl_module.global_step
        )
        pl_module.logger.experiment.add_scalar(
            "recalls/ir_r5", ir_r5, pl_module.global_step
        )
        pl_module.logger.experiment.add_scalar(
            "recalls/ir_r10", ir_r10, pl_module.global_step
        )
        pl_module.logger.experiment.add_scalar(
            "recalls/tr_r1", tr_r1, pl_module.global_step
        )
        pl_module.logger.experiment.add_scalar(
            "recalls/tr_r5", tr_r5, pl_module.global_step
        )
        pl_module.logger.experiment.add_scalar(
            "recalls/tr_r10", tr_r10, pl_module.global_step
        )
        the_metric += ir_r1.item() + tr_r1.item()

    # add for ind irtr
    if pl_module.hparams.config["get_ind_recall_metric"] and not pl_module.training:
        (ir_r1, ir_r5, ir_r10, tr_r1, tr_r5, tr_r10) = compute_ind_irtr_recall(pl_module)
        print((ir_r1, ir_r5, ir_r10, tr_r1, tr_r5, tr_r10), pl_module.global_step)
        pl_module.logger.experiment.add_scalar(
            "recalls/ir_r1", ir_r1, pl_module.global_step
        )
        pl_module.logger.experiment.add_scalar(
            "recalls/ir_r5", ir_r5, pl_module.global_step
        )
        pl_module.logger.experiment.add_scalar(
            "recalls/ir_r10", ir_r10, pl_module.global_step
        )
        pl_module.logger.experiment.add_scalar(
            "recalls/tr_r1", tr_r1, pl_module.global_step
        )
        pl_module.logger.experiment.add_scalar(
            "recalls/tr_r5", tr_r5, pl_module.global_step
        )
        pl_module.logger.experiment.add_scalar(
            "recalls/tr_r10", tr_r10, pl_module.global_step
        )
        # the_metric += ir_r1.item() + tr_r1.item()
        the_metric += ir_r1 + tr_r1
    # == end

    if phase == "val" and len(pl_module.hparams.config["val_datasets"]) > 0:
        # We are using a special dataset for val
        loss_names = pl_module.hparams.config["val_loss_names"]
    else:
        loss_names = pl_module.hparams.config["loss_names"]

    for loss_name, v in loss_names.items():
        if v < 1:
            continue
        if loss_name == "mim" and not hasattr(pl_module, "visual_decoder"):
            # We may choose not to decode
            continue

        value = 0
        qar_value = 0
        if loss_name == "vqa":
            value = getattr(pl_module, f"{phase}_{loss_name}_score").compute()
            pl_module.log(f"{loss_name}/{phase}/score_epoch", value, sync_dist=True)
            getattr(pl_module, f"{phase}_{loss_name}_score").reset()
            pl_module.log(
                f"{loss_name}/{phase}/loss_epoch",
                getattr(pl_module, f"{phase}_{loss_name}_loss").compute(), sync_dist=True
            )
            getattr(pl_module, f"{phase}_{loss_name}_loss").reset()
        elif loss_name == "vcr_q2a":
            # q2a
            pl_module.log(
                f"{loss_name}/{phase}/loss_epoch",
                getattr(pl_module, f"{phase}_{loss_name}_loss").compute(), sync_dist=True
            )
            getattr(pl_module, f"{phase}_{loss_name}_loss").reset()
            value = getattr(pl_module, f"{phase}_{loss_name}_accuracy").compute()
            pl_module.log(f"{loss_name}/{phase}/accuracy_epoch", value, sync_dist=True)
            getattr(pl_module, f"{phase}_{loss_name}_accuracy").reset()
            # qar
            pl_module.log(
                f"vcr_qar/{phase}/loss_epoch",
                getattr(pl_module, f"{phase}_vcr_qar_loss").compute(), sync_dist=True
            )
            getattr(pl_module, f"{phase}_vcr_qar_loss").reset()
            qar_value = getattr(pl_module, f"{phase}_vcr_qar_accuracy").compute()
            pl_module.log(f"vcr_qar/{phase}/accuracy_epoch", qar_value, sync_dist=True)
            getattr(pl_module, f"{phase}_vcr_qar_accuracy").reset()
        # mc_vqa
        elif loss_name == "mc_vqa":
            pl_module.log(
                f"{loss_name}/{phase}/loss_epoch",
                getattr(pl_module, f"{phase}_{loss_name}_loss").compute(),
                sync_dist=True
            )
            getattr(pl_module, f"{phase}_{loss_name}_loss").reset()
            value = getattr(pl_module, f"{phase}_{loss_name}_accuracy").compute()
            pl_module.log(f"{loss_name}/{phase}/accuracy_epoch", value, sync_dist=True)
            getattr(pl_module, f"{phase}_{loss_name}_accuracy").reset()
        elif loss_name == "openend_vqa":
            pl_module.log(
                f"{loss_name}/{phase}/loss_epoch",
                getattr(pl_module, f"{phase}_vqa_loss").compute(),
                sync_dist=True,
            )
            getattr(pl_module, f"{phase}_vqa_loss").reset()
            value = getattr(pl_module, f"{phase}_{loss_name}_accuracy").compute()
            pl_module.log(f"{loss_name}/{phase}/accuracy_epoch", value, sync_dist=True)
            getattr(pl_module, f"{phase}_{loss_name}_accuracy").reset()
        # multiple_choice
        elif loss_name == "multiple_choice":
            pl_module.log(
                f"{loss_name}/{phase}/loss_epoch",
                getattr(pl_module, f"{phase}_{loss_name}_loss").compute(),
                sync_dist=True,
            )
            getattr(pl_module, f"{phase}_{loss_name}_loss").reset()
            value = getattr(pl_module, f"{phase}_{loss_name}_accuracy").compute()
            pl_module.log(f"{loss_name}/{phase}/accuracy_epoch", value, sync_dist=True)
            getattr(pl_module, f"{phase}_{loss_name}_accuracy").reset()
        # vcop
        elif loss_name == "vcop":
            pl_module.log(
                f"{loss_name}/{phase}/loss_epoch",
                getattr(pl_module, f"{phase}_{loss_name}_loss").compute(),
                sync_dist=True,
            )

            getattr(pl_module, f"{phase}_{loss_name}_loss").reset()
            value = getattr(pl_module, f"{phase}_{loss_name}_accuracy").compute()
            pl_module.log(f"{loss_name}/{phase}/accuracy_epoch", value, sync_dist=True)
            getattr(pl_module, f"{phase}_{loss_name}_accuracy").reset()
        elif loss_name == "nlvr2":
            if phase == "train":
                value = getattr(pl_module, f"train_{loss_name}_accuracy").compute()
                pl_module.log(f"{loss_name}/train/accuracy_epoch", value, sync_dist=True)
                getattr(pl_module, f"train_{loss_name}_accuracy").reset()
                pl_module.log(
                    f"{loss_name}/train/loss_epoch",
                    getattr(pl_module, f"train_{loss_name}_loss").compute(),
                    sync_dist=True,
                )
                getattr(pl_module, f"train_{loss_name}_loss").reset()
            else:
                value = getattr(pl_module, f"dev_{loss_name}_accuracy").compute()
                pl_module.log(f"{loss_name}/dev/accuracy_epoch", value, sync_dist=True)
                getattr(pl_module, f"dev_{loss_name}_accuracy").reset()
                pl_module.log(
                    f"{loss_name}/dev/loss_epoch",
                    getattr(pl_module, f"dev_{loss_name}_loss").compute(),
                    sync_dist=True,
                )
                getattr(pl_module, f"dev_{loss_name}_loss").reset()

                value = getattr(pl_module, f"test_{loss_name}_accuracy").compute()
                pl_module.log(f"{loss_name}/test/accuracy_epoch", value, sync_dist=True)
                getattr(pl_module, f"test_{loss_name}_accuracy").reset()
                pl_module.log(
                    f"{loss_name}/test/loss_epoch",
                    getattr(pl_module, f"test_{loss_name}_loss").compute(),
                    sync_dist=True,
                )
                getattr(pl_module, f"test_{loss_name}_loss").reset()
        elif loss_name == "irtr":
            pl_module.log(
                f"{loss_name}/{phase}/irtr_loss_epoch",
                getattr(pl_module, f"{phase}_irtr_loss").compute(), sync_dist=True
            )
            getattr(pl_module, f"{phase}_irtr_loss").reset()
        elif loss_name == "mppd" or loss_name == "mpfr":
            pl_module.log(
                f"{loss_name}/{phase}/loss_epoch",
                getattr(pl_module, f"{phase}_{loss_name}_loss").compute(),
                sync_dist=True,
            )
            getattr(pl_module, f"{phase}_{loss_name}_loss").reset()
        elif loss_name == "vtm":
            value = getattr(pl_module, f"{phase}_{loss_name}_accuracy").compute()
            pl_module.log(f"{loss_name}/{phase}/accuracy_epoch", value, sync_dist=True)
            getattr(pl_module, f"{phase}_{loss_name}_accuracy").reset()
            pl_module.log(
                f"{loss_name}/{phase}/loss_epoch",
                getattr(pl_module, f"{phase}_{loss_name}_loss").compute(),
                sync_dist=True,
            )
            getattr(pl_module, f"{phase}_{loss_name}_loss").reset()
            pl_module.log(
                f"{loss_name}/{phase}/wpa_loss_epoch",
                getattr(pl_module, f"{phase}_{loss_name}_wpa_loss").compute(),
                sync_dist=True,
            )
            getattr(pl_module, f"{phase}_{loss_name}_wpa_loss").reset()
            # add dino loss
            pl_module.log(
                f"{loss_name}/{phase}/dino_loss_epoch",
                getattr(pl_module, f"{phase}_{loss_name}_dino_loss").compute(),
                sync_dist=True,
            )
            getattr(pl_module, f"{phase}_{loss_name}_dino_loss").reset()
        elif loss_name == "dino":
            pl_module.log(
                f"{loss_name}/{phase}/loss_epoch",
                getattr(pl_module, f"{phase}_{loss_name}_loss").compute(),
                sync_dist=True,
            )
            # value = f"{loss_name}/{phase}/loss_epoch",
            getattr(pl_module, f"{phase}_{loss_name}_loss").reset()
        elif loss_name == "vtc":
            pl_module.log(
                f"{loss_name}/{phase}/loss_epoch",
                getattr(pl_module, f"{phase}_{loss_name}_loss").compute(),
                sync_dist=True,
            )
            # value = f"{loss_name}/{phase}/loss_epoch",
            getattr(pl_module, f"{phase}_{loss_name}_loss").reset()
        elif loss_name == "contrastive":
            pl_module.log(
                f"{loss_name}/{phase}/loss_epoch",
                getattr(pl_module, f"{phase}_{loss_name}_loss").compute(),
                sync_dist=True,
            )
            pl_module.log(
                f"{loss_name}/{phase}/image_accuracy_epoch",
                getattr(pl_module, f"{phase}_{loss_name}_image_accuracy").compute(),
                sync_dist=True,
            )
            pl_module.log(
                f"{loss_name}/{phase}/text_accuracy_epoch",
                getattr(pl_module, f"{phase}_{loss_name}_text_accuracy").compute(),
                sync_dist=True,
            )
            getattr(pl_module, f"{phase}_{loss_name}_loss").reset()
            getattr(pl_module, f"{phase}_{loss_name}_image_accuracy").reset()
            getattr(pl_module, f"{phase}_{loss_name}_text_accuracy").reset()
        elif loss_name == "zs_classify":
            value = getattr(pl_module, f"{phase}_{loss_name}_accuracy").compute()
            pl_module.log(f"{loss_name}/{phase}/accuracy_epoch", value, sync_dist=True)
            getattr(pl_module, f"{phase}_{loss_name}_accuracy").reset()
        elif loss_name == "mim":
            pl_module.log(
                f"{loss_name}/{phase}/loss_epoch",
                getattr(pl_module, f"{phase}_{loss_name}_loss").compute(),
                sync_dist=True,
            )
        else:
            value = getattr(pl_module, f"{phase}_{loss_name}_accuracy").compute()
            pl_module.log(f"{loss_name}/{phase}/accuracy_epoch", value, sync_dist=True)
            getattr(pl_module, f"{phase}_{loss_name}_accuracy").reset()
            pl_module.log(
                f"{loss_name}/{phase}/loss_epoch",
                getattr(pl_module, f"{phase}_{loss_name}_loss").compute(),
                sync_dist=True,
            )
            getattr(pl_module, f"{phase}_{loss_name}_loss").reset()

        if loss_name == "vcr_q2a":
            # print(value, qar_value)
            the_metric += qar_value/2 + value/2
        else:
            the_metric += value

    pl_module.log(f"{phase}/the_metric", the_metric)


def check_non_acc_grad(pl_module):
    if pl_module.token_type_embeddings.weight.grad is None:
        return True
    else:
        grad = pl_module.token_type_embeddings.weight.grad
        return (grad.sum() == 0).item()


def set_task(pl_module):
    phase = "train" if pl_module.training else "val"
    if phase == "val" and len(pl_module.hparams.config["val_datasets"]) > 0:
        # We are using a special dataset for val
        pl_module.current_tasks = [
            k for k, v in pl_module.hparams.config["val_loss_names"].items() if v >= 1
        ]
    else:
        pl_module.current_tasks = [
            k for k, v in pl_module.hparams.config["loss_names"].items() if v >= 1
        ]
    return


def set_schedule(pl_module):
    lr = pl_module.hparams.config["learning_rate"]
    wd = pl_module.hparams.config["weight_decay"]

    def no_weight_decay(n, p):
        no_decay = [
            "bias",
            "LayerNorm.bias",
            "LayerNorm.weight",
            "norm.bias",
            "norm.weight",
            "norm1.bias",
            "norm1.weight",
            "norm2.bias",
            "norm2.weight",
            # Following are for clip
            "ln_1.weight",
            "ln_2.weight",
            "ln_3.weight",
            "ln_post.weight",
            "ln_pre.weight",
            "ln_final.weight",
            "embedding",
            "logit_scale",
            # evl
            "temporal_cls_token",
            "pemb_t",
            "input_lns"
        ]
        return p.dim() < 2 or any(nd in n for nd in no_decay)


    head_names = ["vqa_classifier", "nlvr2_classifier"]
    lr_mult = pl_module.hparams.config["lr_mult"]
    clip_lr_mult = pl_module.hparams.config["clip_lr_mult"]
    end_lr = pl_module.hparams.config["end_lr"]
    decay_power = pl_module.hparams.config["decay_power"]
    optim_type = pl_module.hparams.config["optim_type"]

    names = [n for n, p in pl_module.named_parameters()]
    optimizer_grouped_parameters = [
        {
            "params": [
                p
                for n, p in pl_module.named_parameters()
                if not no_weight_decay(n, p)
                and not any(bb in n for bb in head_names)
                and not any(cp in n for cp in clip_param_keys)
            ],
            "weight_decay": wd,
            "lr": lr,
        },
        {
            "params": [
                p
                for n, p in pl_module.named_parameters()
                if no_weight_decay(n, p)
                and not any(bb in n for bb in head_names)
                and not any(cp in n for cp in clip_param_keys)
            ],
            "weight_decay": 0.0,
            "lr": lr,
        },
        {
            "params": [
                p
                for n, p in pl_module.named_parameters()
                if not no_weight_decay(n, p)
                and any(bb in n for bb in head_names)
                and not any(cp in n for cp in clip_param_keys)
            ],
            "weight_decay": wd,
            "lr": lr * lr_mult,
        },
        {
            "params": [
                p
                for n, p in pl_module.named_parameters()
                if no_weight_decay(n, p) and any(bb in n for bb in head_names)
                and not any(cp in n for cp in clip_param_keys)
            ],
            "weight_decay": 0.0,
            "lr": lr * lr_mult,
        },
        {
            "params": [
                p
                for n, p in pl_module.named_parameters()
                if not no_weight_decay(n, p)
                and not any(bb in n for bb in head_names)
                and any(cp in n for cp in clip_param_keys)
            ],
            "weight_decay": wd,
            "lr": lr * clip_lr_mult,
        },
        {
            "params": [
                p
                for n, p in pl_module.named_parameters()
                if no_weight_decay(n, p)
                and not any(bb in n for bb in head_names)
                and any(cp in n for cp in clip_param_keys)
            ],
            "weight_decay": 0.0,
            "lr": lr * clip_lr_mult,
        },
    ]

    if optim_type == "adamw":
        optimizer = torch.optim.AdamW(
            optimizer_grouped_parameters, lr=lr, eps=1e-6, betas=(0.9, 0.98))
    elif optim_type == "adam":
        optimizer = torch.optim.Adam(optimizer_grouped_parameters, lr=lr)
    elif optim_type == "sgd":
        optimizer = torch.optim.SGD(optimizer_grouped_parameters, lr=lr, momentum=0.9)

    if pl_module.trainer.max_steps is None or pl_module.trainer.max_steps == -1:
        max_steps = (
            len(pl_module.trainer.datamodule.train_dataloader())
            * pl_module.trainer.max_epochs
            // pl_module.trainer.accumulate_grad_batches
        )
    else:
        max_steps = pl_module.trainer.max_steps

    warmup_steps = pl_module.hparams.config["warmup_steps"]
    if isinstance(pl_module.hparams.config["warmup_steps"], float):
        warmup_steps = int(max_steps * warmup_steps)

    if decay_power == "cosine":
        scheduler = get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=max_steps,
        )
    else:
        scheduler = get_polynomial_decay_schedule_with_warmup(
            optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=max_steps,
            lr_end=end_lr,
            power=decay_power,
        )
    sched = {"scheduler": scheduler, "interval": "step"}

    return (
        [optimizer],
        [sched],
    )


def state_dict_data_parallel_fix(load_state_dict, curr_state_dict):
    load_keys = list(load_state_dict.keys())
    curr_keys = list(curr_state_dict.keys())

    redo_dp = False
    undo_dp = False
    if not curr_keys[0].startswith('module.') and load_keys[0].startswith('module.'):
        undo_dp = True
    elif curr_keys[0].startswith('module.') and not load_keys[0].startswith('module.'):
        redo_dp = True

    if undo_dp:
        from collections import OrderedDict
        new_state_dict = OrderedDict()
        for k, v in load_state_dict.items():
            name = k[7:]  # remove `module.`
            new_state_dict[name] = v
        # load params
    elif redo_dp:
        from collections import OrderedDict
        new_state_dict = OrderedDict()
        for k, v in load_state_dict.items():
            name = 'module.' + k  # remove `module.`
            new_state_dict[name] = v
    else:
        new_state_dict = load_state_dict
    return new_state_dict


def state_dict_dino_fix(load_state_dict, curr_state_dict):
    load_keys = list(load_state_dict.keys())
    curr_keys = list(curr_state_dict.keys())

    # for k in curr_state_dict.keys():
    #     print(k)
    print('*'*50)
    redo_dp = False
    undo_dp = False
    dino_dp = False
    if not curr_keys[0].startswith('module.') and load_keys[0].startswith('module.'):
        undo_dp = True
    elif curr_keys[0].startswith('module.') and not load_keys[0].startswith('module.'):
        redo_dp = True
    elif load_keys[10].startswith('teacher.') or load_keys[10].startswith('student.'):
        dino_dp = True
    if undo_dp:
        from collections import OrderedDict
        new_state_dict = OrderedDict()
        for k, v in load_state_dict.items():
            # print(k)
            name = k[7:]  # remove `module.`
            new_state_dict[name] = v
        # load params
    elif redo_dp:
        from collections import OrderedDict
        new_state_dict = OrderedDict()
        for k, v in load_state_dict.items():
            # print(k)
            name = 'module.' + k  # remove `module.`
            new_state_dict[name] = v
    elif dino_dp:
        from collections import OrderedDict
        new_state_dict = OrderedDict()
        for k, v in load_state_dict.items():
            # print(k)
            if k[:8] == "student.":
                name = "transformer." + k[8:]  # remove `student.`
                new_state_dict[name] = v
                # continue
            elif k[:8] == "teacher.":
                # name = "transformer." + k[8:]  # remove `teacher.`
                # new_state_dict[name] = v
                continue
            else:
                new_state_dict[k] = v
    else:
        for k, v in load_state_dict.items():
            print(k)
        new_state_dict = load_state_dict
    print('*'*30)
    print("new state dict")
    print('*'*30)
    for k, v in new_state_dict.items():
        print(k)
    return new_state_dict


def read_load_path(load_path):
    if "s3://" in load_path:
        assert client is not None, "Failed to init petrel client"
        model_bytes = client.get(load_path)
        assert load_path is not None, "Read fail from {}".format(load_path)
        return io.BytesIO(model_bytes)
    else:
        return load_path