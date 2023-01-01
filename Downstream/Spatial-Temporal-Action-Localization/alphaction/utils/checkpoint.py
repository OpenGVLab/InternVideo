# Modified from https://github.com/facebookresearch/maskrcnn-benchmark/blob/master/maskrcnn_benchmark/utils/checkpoint.py
import logging
import os

import torch

from alphaction.utils.model_serialization import load_state_dict
from alphaction.utils.c2_model_loading import load_c2_format
from alphaction.structures.memory_pool import MemoryPool


class Checkpointer(object):
    def __init__(
            self,
            model,
            optimizer=None,
            scheduler=None,
            save_dir="",
            save_to_disk=None,
            logger=None,
    ):
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.save_dir = save_dir
        self.save_to_disk = save_to_disk
        if logger is None:
            logger = logging.getLogger(__name__)
        self.logger = logger

    def save(self, name, **kwargs):
        if not self.save_dir:
            return

        if not self.save_to_disk:
            return

        data = {}
        data["model"] = self.model.state_dict()
        if self.optimizer is not None:
            data["optimizer"] = self.optimizer.state_dict()
        if self.scheduler is not None:
            data["scheduler"] = self.scheduler.state_dict()
        data.update(kwargs)

        save_file = os.path.join(self.save_dir, "{}.pth".format(name))
        self.logger.info("Saving checkpoint to {}".format(save_file))
        torch.save(data, save_file)
        self.tag_last_checkpoint(save_file)

    def load(self, f=None, model_weight_only=False, adjust_scheduler=False, no_head=False):
        if self.has_checkpoint():
            # override argument with existing checkpoint
            f = self.get_checkpoint_file()
        if not f:
            # no checkpoint could be found
            self.logger.info("No checkpoint found. Initializing model from scratch")
            return {}
        self.logger.info("Loading checkpoint from {}".format(f))
        checkpoint = self._load_file(f)
        self._load_model(checkpoint, no_head)
        if "optimizer" in checkpoint and self.optimizer:
            if model_weight_only:
                del checkpoint['optimizer']
            else:
                self.logger.info("Loading optimizer from {}".format(f))
                self.optimizer.load_state_dict(checkpoint.pop("optimizer"))
        if "scheduler" in checkpoint and self.scheduler:
            if model_weight_only:
                del checkpoint['scheduler']
            elif adjust_scheduler:
                last_epoch = checkpoint.pop("scheduler")['last_epoch']
                self.logger.info("Adjust scheduler at iteration {}".format(last_epoch))
                self.scheduler.step(last_epoch)
            else:
                self.logger.info("Loading scheduler from {}".format(f))
                self.scheduler.load_state_dict(checkpoint.pop("scheduler"))

        if model_weight_only:
            checkpoint["iteration"] = 0
            checkpoint["person_pool"] = MemoryPool()
        # return any further checkpoint dataset
        return checkpoint

    def has_checkpoint(self):
        save_file = os.path.join(self.save_dir, "last_checkpoint")
        return os.path.exists(save_file)

    def get_checkpoint_file(self):
        save_file = os.path.join(self.save_dir, "last_checkpoint")
        try:
            with open(save_file, "r") as f:
                last_saved = f.read()
                last_saved = last_saved.strip()
        except IOError:
            # if file doesn't exist, maybe because it has just been
            # deleted by a separate process
            last_saved = ""
        return last_saved

    def tag_last_checkpoint(self, last_filename):
        save_file = os.path.join(self.save_dir, "last_checkpoint")
        with open(save_file, "w") as f:
            f.write(last_filename)

    def _load_file(self, f):
        return torch.load(f, map_location=torch.device("cpu"))

    def _load_model(self, checkpoint, no_head):
        load_state_dict(self.model, checkpoint.pop("model"), no_head)


class ActionCheckpointer(Checkpointer):
    def __init__(
            self,
            cfg,
            model,
            optimizer=None,
            scheduler=None,
            save_dir="",
            save_to_disk=None,
            logger=None,
    ):
        super(ActionCheckpointer, self).__init__(
            model, optimizer, scheduler, save_dir, save_to_disk, logger
        )
        self.cfg = cfg.clone()

    def _load_file(self, f):
        if f.endswith(".pkl"):
            return load_c2_format(f, self._get_c2_weight_map())
        loaded = super(ActionCheckpointer, self)._load_file(f)
        if "model" not in loaded:
            loaded = dict(model=loaded)
        return loaded

    def _get_c2_weight_map(self):
        if hasattr(self.model, "c2_weight_mapping"):
            return self.model.c2_weight_mapping()
        elif hasattr(self.model, "module") and hasattr(self.model.module, "c2_weight_mapping"):
            return self.model.module.c2_weight_mapping()
        else:
            raise RuntimeError("Cannot get C2 weight mapping from current model definition.")