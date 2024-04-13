import torch
from fvcore.nn import FlopCountAnalysis, flop_count_table
from torch.nn import MultiheadAttention

from models.beit.st_beit import BeitConfig, BeitModel
from models.temporal_model import (STAdapter, TemporalAttention,
                                   WindowTemporalAttention)


def mem_stat():
    mem = torch.cuda.max_memory_allocated() / 1024 / 1024
    print(f"max memory allocated: {mem}MB")


def build_backbone(tm_block="timesformer"):
    """TODO: Docstring for build_backbone.
    Returns: TODO

    """
    if tm_block == "timesformer":
        other_cfg = dict(
            num_frames=12, temporal_model_block="timesformer", temporal_model_config={}
        )
    elif tm_block == "st_adapter":
        other_cfg = dict(
            num_frames=12, temporal_model_block="st_adapter", temporal_model_config={}
        )
    elif tm_block == "xclip":
        other_cfg = dict(
            num_frames=12, temporal_model_block="xclip", temporal_model_config={}
        )
    elif tm_block == "none":
        other_cfg = dict(num_frames=12, temporal_model_block="none", temporal_model_config={})
    elif tm_block == "wa_2x2":
        other_cfg = dict(
            num_frames=12,
            temporal_model_block="window_attention",
            temporal_model_config=dict(window_size=(2, 2)),
        )
    elif tm_block == "wa_7x7":
        other_cfg = dict(
            num_frames=12,
            temporal_model_block="window_attention",
            temporal_model_config=dict(window_size=(7, 7)),
        )
    else:
        raise ValueError("not exist")

    model_card = "microsoft/beit-base-patch16-224-pt22k-ft22k"
    model_config = BeitConfig.from_pretrained(model_card, image_size=224, **other_cfg)
    model = BeitModel(model_config)
    return model


# model = TemporalAttention()
model = build_backbone("st_adapter")
model.gradient_checkpointing_enable()
model.cuda()
for i in range(3):
    x = torch.rand(32, 12, 3, 224, 224, requires_grad=True)
    x = x.cuda()
    x = x.requires_grad_()
    y = model(x)
    loss = y[0].mean()
    loss.backward()
    mem_stat()

# flops = FlopCountAnalysis(model, x)
# print(flop_count_table(flops))
