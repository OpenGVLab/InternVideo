#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

"""Optimizer."""

import torch

import slowfast.utils.lr_policy as lr_policy
import slowfast.utils.logging as logging

logger = logging.get_logger(__name__)


def construct_optimizer(model, cfg):
    """
    Construct a stochastic gradient descent or ADAM optimizer with momentum.
    Details can be found in:
    Herbert Robbins, and Sutton Monro. "A stochastic approximation method."
    and
    Diederik P.Kingma, and Jimmy Ba.
    "Adam: A Method for Stochastic Optimization."

    Args:
        model (model): model to perform stochastic gradient descent
        optimization or ADAM optimization.
        cfg (config): configs of hyper-parameters of SGD or ADAM, includes base
        learning rate,  momentum, weight_decay, dampening, and etc.
    """
    bn_parameters = []
    non_bn_parameters = []
    zero_parameters = []
    special_bn_parameters = []
    special_non_bn_parameters = []
    special_zero_parameters = []
    skip = {}
    if hasattr(model, "no_weight_decay"):
        skip = model.no_weight_decay()

    special_list = cfg.SOLVER.SPECIAL_LIST
    special_ration = cfg.SOLVER.SPECIAL_RATIO
    logger.info(f'Special parameter list: {special_list}')
    logger.info(f'LR Ration for special parameter is {special_ration}')

    total_num = 0
    for name, m in model.named_modules():
        is_bn = isinstance(m, torch.nn.modules.batchnorm._NormBase)
        for p in m.parameters(recurse=False):
            if not p.requires_grad:
                continue
            total_num += 1
            if is_bn:
                flag = False
                for s in special_list:
                    if s in name:
                        special_bn_parameters.append(p)
                        flag = True
                        break
                if not flag:
                    bn_parameters.append(p)
            elif name in skip or (
                (len(p.shape) == 1 or name.endswith(".bias"))
                and cfg.SOLVER.ZERO_WD_1D_PARAM
            ):
                flag = False
                for s in special_list:
                    if s in name:
                        special_zero_parameters.append(p)
                        flag = True
                        break
                if not flag:
                    zero_parameters.append(p)
            else:
                flag = False
                for s in special_list:
                    if s in name:
                        special_non_bn_parameters.append(p)
                        flag = True
                        break
                if not flag:
                    non_bn_parameters.append(p)

    optim_params = [
        {"params": bn_parameters, "weight_decay": cfg.BN.WEIGHT_DECAY},
        {"params": non_bn_parameters, "weight_decay": cfg.SOLVER.WEIGHT_DECAY},
        {"params": zero_parameters, "weight_decay": 0.0},
        {"params": special_bn_parameters, "weight_decay": cfg.BN.WEIGHT_DECAY, 'lr': cfg.SOLVER.BASE_LR * cfg.SOLVER.SPECIAL_RATIO},
        {"params": special_non_bn_parameters, "weight_decay": cfg.SOLVER.WEIGHT_DECAY, 'lr': cfg.SOLVER.BASE_LR * cfg.SOLVER.SPECIAL_RATIO},
        {"params": special_zero_parameters, "weight_decay": 0.0, 'lr': cfg.SOLVER.BASE_LR * cfg.SOLVER.SPECIAL_RATIO},
    ]
    optim_params = [x for x in optim_params if len(x["params"])]

    # Check all parameters will be passed into optimizer.
    assert total_num == len(non_bn_parameters) + len(special_non_bn_parameters) + \
                        len(bn_parameters) + len(special_bn_parameters) + \
                        len(zero_parameters) + len(special_zero_parameters), \
    "parameter size does not match: {} + {} + {} != {}".format(
        len(non_bn_parameters) + len(+ special_non_bn_parameters),
        len(bn_parameters) + len(special_bn_parameters),
        len(zero_parameters) + len(special_zero_parameters),
        total_num,
    )
    logger.info(
        "BACKBONE: bn {}, non bn {}, zero {}, lr {}".format(
            len(bn_parameters), len(non_bn_parameters), len(zero_parameters),
            cfg.SOLVER.BASE_LR
        )
    )
    logger.info(
        "SPECIAL: bn {}, non bn {}, zero {}, lr {}".format(
            len(special_bn_parameters), len(special_non_bn_parameters), len(special_zero_parameters),
            cfg.SOLVER.BASE_LR * cfg.SOLVER.SPECIAL_RATIO
        )
    )

    if cfg.SOLVER.OPTIMIZING_METHOD == "sgd":
        return torch.optim.SGD(
            optim_params,
            lr=cfg.SOLVER.BASE_LR,
            momentum=cfg.SOLVER.MOMENTUM,
            weight_decay=cfg.SOLVER.WEIGHT_DECAY,
            dampening=cfg.SOLVER.DAMPENING,
            nesterov=cfg.SOLVER.NESTEROV,
        )
    elif cfg.SOLVER.OPTIMIZING_METHOD == "adam":
        return torch.optim.Adam(
            optim_params,
            lr=cfg.SOLVER.BASE_LR,
            betas=(0.9, 0.999),
            weight_decay=cfg.SOLVER.WEIGHT_DECAY,
        )
    elif cfg.SOLVER.OPTIMIZING_METHOD == "adamw":
        return torch.optim.AdamW(
            optim_params,
            lr=cfg.SOLVER.BASE_LR,
            eps=1e-08,
            weight_decay=cfg.SOLVER.WEIGHT_DECAY,
        )
    else:
        raise NotImplementedError(
            "Does not support {} optimizer".format(cfg.SOLVER.OPTIMIZING_METHOD)
        )


def get_epoch_lr(cur_epoch, cfg):
    """
    Retrieves the lr for the given epoch (as specified by the lr policy).
    Args:
        cfg (config): configs of hyper-parameters of ADAM, includes base
        learning rate, betas, and weight decay.
        cur_epoch (float): the number of epoch of the current training stage.
    """
    return lr_policy.get_lr_at_epoch(cfg, cur_epoch)


def set_lr(optimizer, new_lr):
    """
    Sets the optimizer lr to the specified value.
    Args:
        optimizer (optim): the optimizer using to optimize the current network.
        new_lr (float): the new learning rate to set.
    """
    for param_group in optimizer.param_groups:
        param_group["lr"] = new_lr
