from requests import patch
import torch
import torch.nn as nn

from .coca import Residual, ParallelTransformerBlock, CrossAttention
from einops import repeat
import torch.utils.checkpoint as checkpoint
import numpy as np

from timm.models.layers import trunc_normal_ as __call_trunc_normal_


def trunc_normal_(tensor, mean=0.0, std=1.0):
    __call_trunc_normal_(tensor, mean=mean, std=std, a=-std, b=std)


class CaptionDecoder(nn.Module):
    def __init__(
        self,
        n_layers,
        transformer_width,
        vision_width,
        transformer_heads,
        vocab_size,
        num_visual_queries=256,
        use_checkpoint=False,
        checkpoint_num=0,
    ):
        super().__init__()
        scale = transformer_width**-0.5
        self.visual_queries = nn.Parameter(
            scale * torch.randn(num_visual_queries, transformer_width)
        )
        dim_head = transformer_width // transformer_heads
        ff_mult = 4

        self.visual_attn_pooler = CrossAttention(
            dim=transformer_width,
            context_dim=vision_width,
            dim_head=dim_head,
            heads=transformer_heads,
            norm_context=True,
        )
        self.visual_pooler_norm = nn.LayerNorm(transformer_width)

        self.text_norm = nn.LayerNorm(transformer_width)

        self.multimodal_layers = nn.ModuleList([])
        for ind in range(n_layers):
            self.multimodal_layers.append(
                nn.ModuleList(
                    [
                        Residual(
                            ParallelTransformerBlock(
                                dim=transformer_width,
                                dim_head=dim_head,
                                heads=transformer_heads,
                                ff_mult=ff_mult,
                            )
                        ),
                        Residual(
                            CrossAttention(
                                dim=transformer_width,
                                dim_head=dim_head,
                                heads=transformer_heads,
                                parallel_ff=True,
                                ff_mult=ff_mult,
                            )
                        ),
                    ]
                )
            )

        self.predictor = nn.Sequential(
            nn.LayerNorm(transformer_width),
            nn.Linear(transformer_width, transformer_width),
            nn.GELU(),
            nn.LayerNorm(transformer_width),
            nn.Linear(transformer_width, vocab_size),
        )

        self.use_checkpoint = use_checkpoint
        self.checkpoint_num = checkpoint_num

    def forward(self, image_feats, text_embeds):
        # image_feats: # L, N, T, C
        # text_feats: embeded text feats # N, L, C
        # [L, N, T, C] -> [N, T * L, C]
        image_feats = image_feats.permute(1, 0, 2, 3).flatten(1, 2)
        visual_queries = repeat(
            self.visual_queries, 'n d -> b n d', b=image_feats.shape[0]
        )
        image_feats = self.visual_pooler_norm(
            self.visual_attn_pooler(visual_queries, image_feats)
        )

        text_embeds = self.text_norm(text_embeds)
        # go through multimodal layers
        for i, (attn_ff, cross_attn) in enumerate(self.multimodal_layers):
            if self.use_checkpoint and i < self.checkpoint_num:
                text_embeds = checkpoint.checkpoint(attn_ff, text_embeds)
                text_embeds = checkpoint.checkpoint(
                    cross_attn, text_embeds, image_feats
                )
            else:
                text_embeds = attn_ff(text_embeds)
                text_embeds = cross_attn(text_embeds, image_feats)

        logits = self.predictor(text_embeds)

        return logits

