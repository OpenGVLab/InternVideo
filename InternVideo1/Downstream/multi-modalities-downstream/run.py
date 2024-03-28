import warnings

from sklearn import ensemble
# This ignore the scheduler warning, see https://github.com/Lightning-AI/lightning/issues/5558
warnings.filterwarnings("ignore", "Detected call of", UserWarning)

import os
import copy
import pytorch_lightning as pl
from CoTrain.config import ex
from CoTrain.modules import CoTrainTransformerSS, CLIP
from CoTrain.datamodules.video.multitask_datamodule import MTDataModule
import datetime
from pytorch_lightning.utilities.cloud_io import get_filesystem
from pytorch_lightning.utilities.cloud_io import load as pl_load

import torch
import numpy as np
from pytorch_lightning.strategies import DDPStrategy
torch.manual_seed(0)


class CustomDDPStrategy(DDPStrategy):
    def configure_ddp(self):
        super().configure_ddp()
        self._model._set_static_graph() # THIS IS THE MAGIC LINE


def deterministic_index_select(x, dim, indices):
    """
    input_tensor: Tensor
    dim: dim
    indices: 1D tensor
    """
    tensor_transpose = torch.transpose(x, 0, dim)
    return tensor_transpose[indices].transpose(dim, 0)

@ex.automain
def main(_config):
    _config = copy.deepcopy(_config)
    pl.seed_everything(_config["seed"])

    dm = MTDataModule(_config, dist=True)
    if not _config["clip"]:
        model = CoTrainTransformerSS(_config)
    else:
        model = CLIP(_config)
    
    # assert False, [n for n, p in model.named_parameters() if p.requires_grad][:10]

    exp_name = f'{_config["exp_name"]}'

    os.makedirs(_config["log_dir"], exist_ok=True)
    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        save_top_k=_config["save_top_k"],
        # every_n_epochs=_config["save_checkpoints_interval"],
        every_n_train_steps=_config["val_check_interval"],
        verbose=True,
        monitor="contrastive/train/loss",
        mode="min",
        save_last=_config["save_last"],
        dirpath=_config["model_dir"],
    )
    now = datetime.datetime.now()
    if not isinstance(_config["load_path"], str):
        instance_name = f'{exp_name}_seed{_config["seed"]}_from_multiple'
    else:
        instance_name = f'{exp_name}_seed{_config["seed"]}_from_{"_".join(_config["load_path"].split("/")[-2:])[:-5]}'
    logger = pl.loggers.TensorBoardLogger(
        _config["log_dir"],
        name=instance_name,
        version="version_0",
    )

    lr_callback = pl.callbacks.LearningRateMonitor(logging_interval="step")

    summary_callback = pl.callbacks.ModelSummary(max_depth=1)

    callbacks = [checkpoint_callback, lr_callback, summary_callback]

    num_gpus = (
        _config["num_gpus"]
        if isinstance(_config["num_gpus"], int)
        else len(_config["num_gpus"])
    )
    # print all config at the begin
    print('='*70+'Config: '+'='*70)
    print(instance_name)
    print(_config)
    print('='*150)

    # notice _config["batch_size"] should be max length for all machines, eg. at least 1024
    grad_steps = _config["batch_size"] // (
        _config["per_gpu_batchsize"] * num_gpus * _config["num_nodes"]
    )
    assert grad_steps > 0, (_config["batch_size"], _config["per_gpu_batchsize"])

    if not _config["clip_use_checkpoint"]:
        # assert not _config["clip_use_checkpoint"], "Do not use gradient accumulation and checkpoint at the same time"
        if _config["loss_names"]["openend_vqa"] >= 1:
            find_unused_paramters = True
        else:
            find_unused_paramters = False
        strategy = DDPStrategy(find_unused_parameters=find_unused_paramters)
    else:
        assert grad_steps == 1
        strategy = CustomDDPStrategy()

    max_steps = _config["max_steps"] if _config["max_steps"] is not None else None
    resume_ckpt = _config["resume_from"]

    if max_steps == None:
        max_steps = -1

    trainer = pl.Trainer(
        devices=_config["num_gpus"],
        num_nodes=_config["num_nodes"],
        precision=_config["precision"],
        accelerator="gpu",
        benchmark=True,
        deterministic=False,
        max_epochs=_config["max_epoch"] if max_steps == -1 else 100,
        max_steps=max_steps,
        callbacks=callbacks,
        logger=logger,
        # prepare_data_per_node=False,
        replace_sampler_ddp=False,
        accumulate_grad_batches=grad_steps,
        log_every_n_steps=10,
        fast_dev_run=_config["fast_dev_run"],
        val_check_interval=_config["val_check_interval"],
        strategy=strategy,
        # show_progress_bar=False,
        # progress_bar_refresh_rate=0
    )

    fs = get_filesystem(resume_ckpt)
    if fs.exists(resume_ckpt):
        with fs.open(resume_ckpt, "rb") as f:
            ckpt = torch.load(f, map_location='cpu')
        # This hacks pl wrong steps for logger
        global_step_offset = ckpt["global_step"]
        trainer.fit_loop.epoch_loop._batches_that_stepped = global_step_offset
        del ckpt
        pass
    else:
        resume_ckpt = None

    print("accumulate grad batches is: ", trainer.accumulate_grad_batches)

    if not _config["test_only"]:
        with torch.autograd.set_detect_anomaly(True):
            trainer.fit(model, datamodule=dm, ckpt_path=resume_ckpt)
    else:
        trainer.test(model, datamodule=dm, ckpt_path=resume_ckpt)
