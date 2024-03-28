#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

"""Argument parser functions."""

import argparse
import sys

import slowfast.utils.checkpoint as cu
from slowfast.config.defaults import get_cfg


def parse_args():
    """
    Parse the following arguments for a default parser for PySlowFast users.
    Args:
        shard_id (int): shard id for the current machine. Starts from 0 to
            num_shards - 1. If single machine is used, then set shard id to 0.
        num_shards (int): number of shards using by the job.
        init_method (str): initialization method to launch the job with multiple
            devices. Options includes TCP or shared file-system for
            initialization. details can be find in
            https://pytorch.org/docs/stable/distributed.html#tcp-initialization
        cfg (str): path to the config file.
        opts (argument): provide addtional options from the command line, it
            overwrites the config loaded from file.
    """
    parser = argparse.ArgumentParser(
        description="Provide SlowFast video training and testing pipeline."
    )
    parser.add_argument(
        "--shard_id",
        help="The shard id of current node, Starts from 0 to num_shards - 1",
        default=0,
        type=int,
    )
    parser.add_argument(
        "--num_shards",
        help="Number of shards using by the job",
        default=1,
        type=int,
    )
    parser.add_argument(
        "--init_method",
        help="Initialization method, includes TCP or shared file-system",
        default="tcp://localhost:9999",
        type=str,
    )
    parser.add_argument(
        "--cfg",
        dest="cfg_file",
        help="Path list to the config file",
        nargs='+',
        default=[],
    )
    parser.add_argument(
        "opts",
        help="See slowfast/config/defaults.py for all options",
        default=None,
        nargs=argparse.REMAINDER,
    )
    if len(sys.argv) == 1:
        parser.print_help()
    return parser.parse_args()


def load_config(args):
    """
    Given the arguemnts, load and initialize the configs.
    Args:
        args (argument): arguments includes `shard_id`, `num_shards`,
            `init_method`, `cfg_file`, and `opts`.
    """
    cfg_num = len(args.cfg_file)
    cfg_list = [get_cfg() for _ in range(cfg_num)]
    for i in range(cfg_num):
        cfg_list[i].merge_from_file(args.cfg_file[i])
        # Load config from command line, overwrite config from opts.
        if args.opts is not None:
            cfg_list[i].merge_from_list(args.opts)

        # Inherit parameters from args.
        if hasattr(args, "num_shards") and hasattr(args, "shard_id"):
            cfg_list[i].NUM_SHARDS = args.num_shards
            cfg_list[i].SHARD_ID = args.shard_id
        if hasattr(args, "rng_seed"):
            cfg_list[i].RNG_SEED = args.rng_seed
        if hasattr(args, "output_dir"):
            cfg_list[i].OUTPUT_DIR = args.output_dir

    # Create the checkpoint dir.
    cu.make_checkpoint_dir(cfg_list[0].OUTPUT_DIR)
    return cfg_list
