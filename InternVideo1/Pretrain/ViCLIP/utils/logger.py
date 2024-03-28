# from MMF: https://github.com/facebookresearch/mmf/blob/master/mmf/utils/logger.py
# Copyright (c) Facebook, Inc. and its affiliates.

import functools
import logging
import os
import sys
import time
import wandb
from typing import Any, Dict, Union

import torch
from .distributed import get_rank, is_main_process
from termcolor import colored


def log_dict_to_wandb(log_dict, step, prefix=""):
    """include a separator `/` at the end of `prefix`"""
    if not is_main_process():
        return

    log_dict = {f"{prefix}{k}": v for k, v in log_dict.items()}
    wandb.log(log_dict, step)


def setup_wandb(config):
    if not (config.wandb.enable and is_main_process()):
        return

    run = wandb.init(
        config=config,
        project=config.wandb.project,
        entity=config.wandb.entity,
        name=os.path.basename(config.output_dir),
        reinit=True
    )
    return run


def setup_output_folder(save_dir: str, folder_only: bool = False):
    """Sets up and returns the output file where the logs will be placed
    based on the configuration passed. Usually "save_dir/logs/log_<timestamp>.txt".
    If env.log_dir is passed, logs will be directly saved in this folder.
    Args:
        folder_only (bool, optional): If folder should be returned and not the file.
            Defaults to False.
    Returns:
        str: folder or file path depending on folder_only flag
    """
    log_filename = "train_"
    log_filename += time.strftime("%Y_%m_%dT%H_%M_%S")
    log_filename += ".log"

    log_folder = os.path.join(save_dir, "logs")

    if not os.path.exists(log_folder):
        os.path.mkdirs(log_folder)

    if folder_only:
        return log_folder

    log_filename = os.path.join(log_folder, log_filename)

    return log_filename


def setup_logger(
    output: str = None,
    color: bool = True,
    name: str = "mmf",
    disable: bool = False,
    clear_handlers=True,
    *args,
    **kwargs,
):
    """
    Initialize the MMF logger and set its verbosity level to "INFO".
    Outside libraries shouldn't call this in case they have set there
    own logging handlers and setup. If they do, and don't want to
    clear handlers, pass clear_handlers options.
    The initial version of this function was taken from D2 and adapted
    for MMF.
    Args:
        output (str): a file name or a directory to save log.
            If ends with ".txt" or ".log", assumed to be a file name.
            Default: Saved to file <save_dir/logs/log_[timestamp].txt>
        color (bool): If false, won't log colored logs. Default: true
        name (str): the root module name of this logger. Defaults to "mmf".
        disable: do not use
        clear_handlers (bool): If false, won't clear existing handlers.
    Returns:
        logging.Logger: a logger
    """
    if disable:
        return None
    logger = logging.getLogger(name)
    logger.propagate = False

    logging.captureWarnings(True)
    warnings_logger = logging.getLogger("py.warnings")

    plain_formatter = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(name)s : %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S",
    )

    distributed_rank = get_rank()
    handlers = []

    logging_level = logging.INFO
    # logging_level = logging.DEBUG

    if distributed_rank == 0:
        logger.setLevel(logging_level)
        ch = logging.StreamHandler(stream=sys.stdout)
        ch.setLevel(logging_level)
        if color:
            formatter = ColorfulFormatter(
                colored("%(asctime)s | %(name)s: ", "green") + "%(message)s",
                datefmt="%Y-%m-%dT%H:%M:%S",
            )
        else:
            formatter = plain_formatter
        ch.setFormatter(formatter)
        logger.addHandler(ch)
        warnings_logger.addHandler(ch)
        handlers.append(ch)

    # file logging: all workers
    if output is None:
        output = setup_output_folder()

    if output is not None:
        if output.endswith(".txt") or output.endswith(".log"):
            filename = output
        else:
            filename = os.path.join(output, "train.log")
        if distributed_rank > 0:
            filename = filename + f".rank{distributed_rank}"
        os.makedirs(os.path.dirname(filename), exist_ok=True)

        fh = logging.StreamHandler(_cached_log_stream(filename))
        fh.setLevel(logging_level)
        fh.setFormatter(plain_formatter)
        logger.addHandler(fh)
        warnings_logger.addHandler(fh)
        handlers.append(fh)

        # Slurm/FB output, only log the main process
        #     save_dir = get_mmf_env(key="save_dir")
        if "train.log" not in filename and distributed_rank == 0:
            filename = os.path.join(output, "train.log")
            sh = logging.StreamHandler(_cached_log_stream(filename))
            sh.setLevel(logging_level)
            sh.setFormatter(plain_formatter)
            logger.addHandler(sh)
            warnings_logger.addHandler(sh)
            handlers.append(sh)

        logger.info(f"Logging to: {filename}")

    # Remove existing handlers to add MMF specific handlers
    if clear_handlers:
        for handler in logging.root.handlers[:]:
            logging.root.removeHandler(handler)
    # Now, add our handlers.
    logging.basicConfig(level=logging_level, handlers=handlers)

    return logger


def setup_very_basic_config(color=True):
    plain_formatter = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(name)s : %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S",
    )
    ch = logging.StreamHandler(stream=sys.stdout)
    ch.setLevel(logging.INFO)
    if color:
        formatter = ColorfulFormatter(
            colored("%(asctime)s | %(name)s: ", "green") + "%(message)s",
            datefmt="%Y-%m-%dT%H:%M:%S",
        )
    else:
        formatter = plain_formatter
    ch.setFormatter(formatter)
    # Setup a minimal configuration for logging in case something tries to
    # log a message even before logging is setup by MMF.
    logging.basicConfig(level=logging.INFO, handlers=[ch])


# cache the opened file object, so that different calls to `setup_logger`
# with the same file name can safely write to the same file.
@functools.lru_cache(maxsize=None)
def _cached_log_stream(filename):
    return open(filename, "a")


# ColorfulFormatter is adopted from Detectron2 and adapted for MMF
class ColorfulFormatter(logging.Formatter):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def formatMessage(self, record):
        log = super().formatMessage(record)
        if record.levelno == logging.WARNING:
            prefix = colored("WARNING", "red", attrs=["blink"])
        elif record.levelno == logging.ERROR or record.levelno == logging.CRITICAL:
            prefix = colored("ERROR", "red", attrs=["blink", "underline"])
        else:
            return log
        return prefix + " " + log


class TensorboardLogger:
    def __init__(self, log_folder="./logs", iteration=0):
        # This would handle warning of missing tensorboard
        from torch.utils.tensorboard import SummaryWriter

        self.summary_writer = None
        self._is_master = is_main_process()
        # self.timer = Timer()
        self.log_folder = log_folder

        if self._is_master:
            # current_time = self.timer.get_time_hhmmss(None, format=self.time_format)
            current_time = time.strftime("%Y-%m-%dT%H:%M:%S")
            # self.timer.get_time_hhmmss(None, format=self.time_format)
            tensorboard_folder = os.path.join(
                self.log_folder, f"tensorboard_{current_time}"
            )
            self.summary_writer = SummaryWriter(tensorboard_folder)

    def __del__(self):
        if getattr(self, "summary_writer", None) is not None:
            self.summary_writer.close()

    def _should_log_tensorboard(self):
        if self.summary_writer is None or not self._is_master:
            return False
        else:
            return True

    def add_scalar(self, key, value, iteration):
        if not self._should_log_tensorboard():
            return

        self.summary_writer.add_scalar(key, value, iteration)

    def add_scalars(self, scalar_dict, iteration):
        if not self._should_log_tensorboard():
            return

        for key, val in scalar_dict.items():
            self.summary_writer.add_scalar(key, val, iteration)

    def add_histogram_for_model(self, model, iteration):
        if not self._should_log_tensorboard():
            return

        for name, param in model.named_parameters():
            np_param = param.clone().cpu().data.numpy()
            self.summary_writer.add_histogram(name, np_param, iteration)
