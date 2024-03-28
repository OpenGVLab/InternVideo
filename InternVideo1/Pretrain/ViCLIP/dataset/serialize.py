# Description: This file contains the code for serializing the dataset.
# From https://github.com/ppwwyyxx/RAM-multiprocess-dataloader/blob/795868a37446d61412b9a58dbb1b7c76e75d39c4/serialize.py

# Copyright (c) Facebook, Inc. and its affiliates.
"""
List serialization code adopted from
https://github.com/facebookresearch/detectron2/blob/main/detectron2/data/common.py
"""

import multiprocessing as mp

from typing import List, Any, Optional

import pickle
import numpy as np
import torch
import torch.distributed as dist

import functools
import os

from datetime import timedelta


def get_world_size() -> int:
    if not dist.is_available():
        return 1
    if not dist.is_initialized():
        return 1
    return dist.get_world_size()


def get_rank() -> int:
    if not dist.is_available():
        return 0
    if not dist.is_initialized():
        return 0
    return dist.get_rank()


def get_local_rank() -> int:
    if not dist.is_available():
        return 0
    if not dist.is_initialized():
        return 0
    
    # this is not guaranteed to be set
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        return int(os.environ['LOCAL_RANK'])
    elif 'SLURM_PROCID' in os.environ:
        return int(os.environ['SLURM_LOCALID'])
    else:
        raise RuntimeError("Unable to get local rank")


def get_local_size() -> int:
    return torch.cuda.device_count()


@functools.lru_cache()
def _get_global_gloo_group():
    """
    Return a process group based on gloo backend, containing all the ranks
    The result is cached.
    """
    if dist.get_backend() == "nccl":
        return dist.new_group(backend="gloo", timeout=timedelta(minutes=60))
    else:
        return dist.group.WORLD


def all_gather(data, group=None):
    """
    Run all_gather on arbitrary picklable data (not necessarily tensors).

    Args:
        data: any picklable object
        group: a torch process group. By default, will use a group which
            contains all ranks on gloo backend.

    Returns:
        list[data]: list of data gathered from each rank
    """
    if get_world_size() == 1:
        return [data]
    if group is None:
        group = (
            _get_global_gloo_group()
        )  # use CPU group by default, to reduce GPU RAM usage.
    world_size = dist.get_world_size(group)
    if world_size == 1:
        return [data]

    output = [None for _ in range(world_size)]
    dist.all_gather_object(output, data, group=group)
    return output


class NumpySerializedList:
    def __init__(self, lst: list):
        def _serialize(data):
            buffer = pickle.dumps(data, protocol=-1)
            return np.frombuffer(buffer, dtype=np.uint8)

        print(
            "Serializing {} elements to byte tensors and concatenating them all ...".format(
                len(lst)
            )
        )
        self._lst = [_serialize(x) for x in lst]
        self._addr = np.asarray([len(x) for x in self._lst], dtype=np.int64)
        self._addr = np.cumsum(self._addr)
        self._lst = np.concatenate(self._lst)
        print("Serialized dataset takes {:.2f} MiB".format(len(self._lst) / 1024**2))

    def __len__(self):
        return len(self._addr)

    def __getitem__(self, idx):
        start_addr = 0 if idx == 0 else self._addr[idx - 1].item()
        end_addr = self._addr[idx].item()
        bytes = memoryview(self._lst[start_addr:end_addr])
        return pickle.loads(bytes)


class TorchSerializedList(NumpySerializedList):
    def __init__(self, lst: list):
        super().__init__(lst)
        self._addr = torch.from_numpy(self._addr)
        self._lst = torch.from_numpy(self._lst)

    def __getitem__(self, idx):
        start_addr = 0 if idx == 0 else self._addr[idx - 1].item()
        end_addr = self._addr[idx].item()
        bytes = memoryview(self._lst[start_addr:end_addr].numpy())
        return pickle.loads(bytes)


def local_scatter(array: Optional[List[Any]]):
    """
    Scatter an array from local leader to all local workers.
    The i-th local worker gets array[i].

    Args:
        array: Array with same size of #local workers.
    """
    if get_local_size() <= 1:
        # Just one worker. Do nothing.
        return array[0]
    if get_local_rank() == 0:
        assert len(array) == get_local_size()
        all_gather(array)
    else:
        all_data = all_gather(None)
        array = all_data[get_rank() - get_local_rank()]
    return array[get_local_rank()]


# NOTE: https://github.com/facebookresearch/mobile-vision/pull/120
# has another implementation that does not use tensors.
class TorchShmSerializedList(TorchSerializedList):
    def __init__(self, lst: list):
        if get_local_rank() == 0:
            super().__init__(lst)

        if get_local_rank() == 0:
            # Move data to shared memory, obtain a handle to send to each local worker.
            # This is cheap because a tensor will only be moved to shared memory once.
            handles = [None] + [
                bytes(mp.reduction.ForkingPickler.dumps((self._addr, self._lst)))
                for _ in range(get_local_size() - 1)
            ]
        else:
            handles = None
        # Each worker receives the handle from local leader.
        handle = local_scatter(handles)

        if get_local_rank() > 0:
            # Materialize the tensor from shared memory.
            self._addr, self._lst = mp.reduction.ForkingPickler.loads(handle)
            print(
                f"Worker {get_rank()} obtains a dataset of length="
                f"{len(self)} from its local leader."
            )


# From https://github.com/ppwwyyxx/RAM-multiprocess-dataloader/issues/5#issuecomment-1510676170
def local_broadcast_process_authkey():
    if int(os.environ['LOCAL_WORLD_SIZE']) == 1:
        return 
    local_rank = int(os.environ['LOCAL_RANK'])
    authkey = bytes(mp.current_process().authkey)
    all_keys = all_gather(authkey)
    local_leader_key = all_keys[get_rank() - local_rank]
    if authkey != local_leader_key:
        print("Process authkey is different from the key of local leader. This might happen when "
              "workers are launched independently.")
        print("Overwriting local authkey ...")
        mp.current_process().authkey = local_leader_key
