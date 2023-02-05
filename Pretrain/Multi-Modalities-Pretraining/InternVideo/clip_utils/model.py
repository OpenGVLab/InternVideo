from collections import OrderedDict
from typing import Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.checkpoint import checkpoint_sequential

from . import utils


class LayerNorm(nn.LayerNorm):
    """Subclass torch's LayerNorm to handle fp16."""

    def forward(self, x: torch.Tensor):
        orig_type = x.dtype
        ret = super().forward(x.type(torch.float32))
        return ret.type(orig_type)


class QuickGELU(nn.Module):
    def forward(self, x: torch.Tensor):
        return x * torch.sigmoid(1.702 * x)


class ResidualAttentionBlock(nn.Module):
    def __init__(self, d_model: int, n_head: int, attn_mask: torch.Tensor = None):
        super().__init__()

        self.attn = nn.MultiheadAttention(d_model, n_head)
        self.ln_1 = LayerNorm(d_model)
        self.mlp = nn.Sequential(
            OrderedDict(
                [
                    ("c_fc", nn.Linear(d_model, d_model * 4)),
                    ("gelu", QuickGELU()),
                    ("c_proj", nn.Linear(d_model * 4, d_model)),
                ]
            )
        )
        self.ln_2 = LayerNorm(d_model)
        self.attn_mask = attn_mask

    def attention(self, x: torch.Tensor):
        self.attn_mask = (
            self.attn_mask.to(dtype=x.dtype, device=x.device)
            if self.attn_mask is not None
            else None
        )
        return self.attn(x, x, x, need_weights=False, attn_mask=self.attn_mask)[0]

    def forward(self, x: torch.Tensor):
        x = x + self.attention(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


class Transformer(nn.Module):
    def __init__(
        self,
        width: int,
        layers: int,
        heads: int,
        attn_mask: torch.Tensor = None,
        use_checkpoint=False,
        checkpoint_num=[0, 0],
    ):
        super().__init__()
        self.width = width
        self.layers = layers
        self.resblocks = nn.Sequential(
            *[ResidualAttentionBlock(width, heads, attn_mask) for _ in range(layers)]
        )
        self.use_checkpoint = use_checkpoint
        self.checkpoint_num = checkpoint_num

    def forward(self, x: torch.Tensor):
        if self.use_checkpoint and self.checkpoint_num[1] > 0:
            segments = min(len(self.resblocks), self.checkpoint_num[1])
            return checkpoint_sequential(self.resblocks, segments, x)
        else:
            return self.resblocks(x)


class VideoIntern(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        # vision
        vision_width: int,
        # text
        context_length: int,
        vocab_size: int,
        transformer_width: int,
        transformer_heads: int,
        transformer_layers: int,
        # uni
        n_layers=4,
        n_dim=768,
        n_head=12,
        drop_path_rate=0.0,
        mlp_dropout=[0.5, 0.5, 0.5, 0.5],
        cls_dropout=0.5,
        t_size=8,
        use_image_attnmap=True,
        backbone='vit_2plus1d_dw_bias_b16',
        return_list=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11],
        use_checkpoint=False,
        checkpoint_num=[0],
    ):
        super().__init__()

        self.vision_width = n_dim

        self.context_length = context_length

        self.visual = utils.__dict__[backbone](
            pretrained=False,
            t_size=t_size,
            mlp_dropout=mlp_dropout,
            cls_dropout=cls_dropout,
            n_dim=n_dim,
            n_head=n_head,
            return_list=return_list,
            drop_path_rate=drop_path_rate,
            backbone_drop_path_rate=drop_path_rate,
            use_checkpoint=use_checkpoint,
            checkpoint_num=checkpoint_num,
        )

        self.visual_ln_post = nn.LayerNorm(n_dim)
        scale = n_dim**-0.5
        self.visual_proj = nn.Parameter(scale * torch.randn(n_dim, embed_dim))
        self.return_qk = use_image_attnmap
        self.return_num = n_layers

        self.transformer = Transformer(
            width=transformer_width,
            layers=transformer_layers,
            heads=transformer_heads,
            attn_mask=self.build_attention_mask(),
            use_checkpoint=use_checkpoint,
            checkpoint_num=checkpoint_num,
        )

        self.vocab_size = vocab_size
        self.token_embedding = nn.Embedding(vocab_size, transformer_width)
        self.positional_embedding = nn.Parameter(
            torch.empty(self.context_length, transformer_width)
        )
        self.ln_final = LayerNorm(transformer_width)

        self.text_projection = nn.Parameter(torch.empty(transformer_width, embed_dim))
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

        self.embed_dim = embed_dim

        # We seperate the mask embedding to load pretrained model
        self.text_mask_embedding = nn.Parameter(torch.empty(1, 1, transformer_width))

        # # To keep the num_embeddings unchanged, we add this to embedded text
        # self.eot_token_embedding = nn.Parameter(torch.empty(1, transformer_width))

        self.initialize_parameters()

    def initialize_parameters(self):
        nn.init.normal_(self.token_embedding.weight, std=0.02)
        nn.init.normal_(self.positional_embedding, std=0.01)
        nn.init.normal_(self.text_mask_embedding, std=0.02)
        # nn.init.constant_(self.eot_token_embedding, 0.0)

        proj_std = (self.transformer.width**-0.5) * (
            (2 * self.transformer.layers) ** -0.5
        )
        attn_std = self.transformer.width**-0.5
        fc_std = (2 * self.transformer.width) ** -0.5
        for block in self.transformer.resblocks:
            nn.init.normal_(block.attn.in_proj_weight, std=attn_std)
            nn.init.normal_(block.attn.out_proj.weight, std=proj_std)
            nn.init.normal_(block.mlp.c_fc.weight, std=fc_std)
            nn.init.normal_(block.mlp.c_proj.weight, std=proj_std)

        if self.text_projection is not None:
            nn.init.normal_(self.text_projection, std=self.transformer.width**-0.5)

        nn.init.constant_(self.visual_ln_post.weight, 1.0)
        nn.init.constant_(self.visual_ln_post.bias, 0.0)

    def build_attention_mask(self):
        # lazily create causal attention mask, with full attention between the vision tokens
        # pytorch uses additive attention mask; fill with -inf
        mask = torch.empty(self.context_length, self.context_length)
        mask.fill_(float("-inf"))
        mask.triu_(1)  # zero out the lower diagonal
        return mask

    @property
    def dtype(self):
        return self.visual.conv1.weight.dtype

    def encode_video(
        self, video, return_all_feats=False, masked_indices=None, mode="video"
    ):
        # video: [N, C, T, H, W]
        feats = self.visual(video, return_all_feats=return_all_feats, mode=mode)
        if return_all_feats:
            x, feats = feats
        else:
            x = feats
        x = self.visual_ln_post(x)
        if self.visual_proj is not None:
            x = x @ self.visual_proj

        if return_all_feats:
            return x, feats  # [N, C], [L, N, T, C]

        return x

    def encode_text(self, text, masked_indices=None, return_all_feats=False):
        # assert (text.max(dim=-1)[0] + 1 == self.token_embedding.num_embeddings).all(), \
        #     "The last token of each sentence should be eot_token, check the input"

        x = self.token_embedding(text).type(self.dtype)  # [batch_size, n_ctx, d_model]
        # x[torch.arange(x.shape[0]), text.argmax(dim=-1)] += self.eot_token_embedding

        if masked_indices is not None:
            x[masked_indices] = self.text_mask_embedding

        x = x + self.positional_embedding.type(self.dtype)
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x).type(self.dtype)

        # x.shape = [batch_size, n_ctx, transformer.width]
        # take features from the eot embedding (eot_token is the highest number in each sequence)
        feats = x[torch.arange(x.shape[0]), text.argmax(dim=-1)]
        if self.text_projection is not None:
            feats = feats @ self.text_projection

        if return_all_feats:
            return feats, x

        return feats


def build_model(
    state_dict: dict,
    n_layers=4,
    n_dim=768,
    n_head=12,
    mlp_factor=4.0,
    drop_path_rate=0.0,
    mlp_dropout=[0.5, 0.5, 0.5, 0.5],
    cls_dropout=0.5,
    t_size=8,
    spatial_size=14,
    use_t_conv=True,
    use_image_attnmap=True,
    use_t_pos_embed=True,
    no_pretrain=False,
    init_zero=True,
    use_checkpoint=False,
    checkpoint_num=[0],
):
    vision_width = state_dict["visual.conv1.weight"].shape[0]
    vision_layers = len(
        [
            k
            for k in state_dict.keys()
            if k.startswith("visual.") and k.endswith(".attn.in_proj_weight")
        ]
    )
    vision_patch_size = state_dict["visual.conv1.weight"].shape[-1]
    grid_size = round((state_dict["visual.positional_embedding"].shape[0] - 1) ** 0.5)
    image_resolution = vision_patch_size * grid_size

    embed_dim = state_dict["text_projection"].shape[1]
    context_length = state_dict["positional_embedding"].shape[0]
    vocab_size = state_dict["token_embedding.weight"].shape[0]
    transformer_width = state_dict["ln_final.weight"].shape[0]
    transformer_heads = transformer_width // 64
    transformer_layers = len(
        set(
            k.split(".")[2]
            for k in state_dict
            if k.startswith(f"transformer.resblocks")
        )
    )

    vision_width = state_dict["visual_proj"].shape[0]
    n_dim = vision_width
    if vision_width == 768:
        backbone = "vit_only_global_b16"
        n_head = 12
        return_list = [8, 9, 10, 11]
    elif vision_width == 1024:
        backbone = "vit_only_global_l14"
        n_head = 16
        return_list = [20, 21, 22, 23]
    else:
        raise NotImplementedError

    model = VideoIntern(
        embed_dim,
        vision_width,
        context_length,
        vocab_size,
        transformer_width,
        transformer_heads,
        transformer_layers,
        n_layers=n_layers,
        n_dim=n_dim,
        n_head=n_head,
        drop_path_rate=drop_path_rate,
        mlp_dropout=mlp_dropout,
        cls_dropout=cls_dropout,
        t_size=t_size,
        use_image_attnmap=use_image_attnmap,
        backbone=backbone,
        return_list=return_list,
        use_checkpoint=use_checkpoint,
        checkpoint_num=checkpoint_num,
    )

    model.load_state_dict(state_dict, strict=False)

    return model.eval()
