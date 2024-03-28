# Modified from https://github.com/facebookresearch/maskrcnn-benchmark/blob/master/maskrcnn_benchmark/utils/logger.py
import logging
import os
import sys
import time


def setup_logger(name, save_dir, distributed_rank, filename=None):
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    logger.propagate = False
    # don't log results for the non-master process
    if distributed_rank > 0:
        return logger
    ch = logging.StreamHandler(stream=sys.stdout)
    ch.setLevel(logging.DEBUG)
    formatter = logging.Formatter("%(asctime)s %(name)s %(levelname)s: %(message)s")
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    if save_dir:
        if filename is None:
            filename = time.strftime("%Y-%m-%d_%H.%M.%S", time.localtime()) + ".log"
        fh = logging.FileHandler(os.path.join(save_dir, filename))
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(formatter)
        logger.addHandler(fh)

    return logger

def setup_tblogger(save_dir, distributed_rank):
    if distributed_rank>0:
        return None
    from tensorboardX import SummaryWriter
    tbdir = os.path.join(save_dir,'tb')
    os.makedirs(tbdir,exist_ok=True)
    tblogger = SummaryWriter(tbdir)
    return tblogger