#!/usr/bin/env python

from collections import OrderedDict

from timm.models.layers import DropPath
import torch
import torch.nn as nn
import torch.nn.functional as F


class QuickGELU(nn.Module):
    def forward(self, x: torch.Tensor):
        return x * torch.sigmoid(1.702 * x)


class ResidualDecoderBlock(nn.Module):
    def __init__(self, d_model: int, n_head: int, attn_mask: torch.Tensor = None,
                 mlp_factor: float = 4.0, dropout: float = 0.0, drop_path: float = 0.0):
        super().__init__()

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        print(f'Drop path rate: {drop_path}')
        self.attn = nn.MultiheadAttention(d_model, n_head)
        self.ln_1 = nn.LayerNorm(d_model)
        d_mlp = round(mlp_factor * d_model)
        self.mlp = nn.Sequential(OrderedDict([
            ("c_fc", nn.Linear(d_model, d_mlp)),
            ("gelu", QuickGELU()),
            ("dropout", nn.Dropout(dropout)),
            ("c_proj", nn.Linear(d_mlp, d_model))
        ]))
        self.ln_2 = nn.LayerNorm(d_model)
        self.ln_3 = nn.LayerNorm(d_model)
        self.attn_mask = attn_mask

        nn.init.xavier_uniform_(self.attn.in_proj_weight)
        nn.init.xavier_uniform_(self.attn.out_proj.weight)
        nn.init.xavier_uniform_(self.mlp[0].weight)
        nn.init.xavier_uniform_(self.mlp[-1].weight)

    def attention(self, x: torch.Tensor, y: torch.Tensor):
        #self.attn_mask = self.attn_mask.to(dtype=x.dtype, device=x.device) if self.attn_mask is not None else None
        # return self.attn(x, y, y, need_weights=False, attn_mask=self.attn_mask)[0]
        assert self.attn_mask is None  # not implemented
        # manual forward to add position information
        d_model = self.ln_1.weight.size(0)
        q = (x @ self.attn.in_proj_weight[:d_model].T) + self.attn.in_proj_bias[:d_model]

        k = (y @ self.attn.in_proj_weight[d_model:-d_model].T) + self.attn.in_proj_bias[d_model:-d_model]
        v = (y @ self.attn.in_proj_weight[-d_model:].T) + self.attn.in_proj_bias[-d_model:]
        Tx, Ty, N = q.size(0), k.size(0), q.size(1)
        q = q.view(Tx, N, self.attn.num_heads, self.attn.head_dim).permute(1, 2, 0, 3)
        k = k.view(Ty, N, self.attn.num_heads, self.attn.head_dim).permute(1, 2, 0, 3)
        v = v.view(Ty, N, self.attn.num_heads, self.attn.head_dim).permute(1, 2, 0, 3)
        aff = (q @ k.transpose(-2, -1) / (self.attn.head_dim ** 0.5))

        aff = aff.softmax(dim=-1)
        out = aff @ v
        out = out.permute(2, 0, 1, 3).flatten(2)
        out = self.attn.out_proj(out)
        return out

    def forward(self, x: torch.Tensor, y: torch.Tensor):
        x = x + self.drop_path(self.attention(self.ln_1(x), self.ln_3(y)))
        x = x + self.drop_path(self.mlp(self.ln_2(x)))
        return x


class TransformerDecoder(nn.Module):
    def __init__(self, n_layers=4,
                 n_dim=768, n_head=12, mlp_factor=4.0, drop_path_rate=0.,
                 mlp_dropout=[0.5, 0.5, 0.5, 0.5], cls_dropout=0.5, t_size=8,
                 use_t_conv=True, use_t_pos_embed=True, num_classes=400,
                 add_residual=False,
                 ):
        super().__init__()
        
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, n_layers)]
        self.dec = nn.ModuleList([
            ResidualDecoderBlock(n_dim, n_head, mlp_factor=mlp_factor, dropout=mlp_dropout[i], drop_path=dpr[i])
            for i in range(n_layers)
        ])
        self.proj = nn.Sequential(
            nn.LayerNorm(n_dim),
            nn.Dropout(cls_dropout),
            nn.Linear(n_dim, num_classes),
        )
        self.temporal_cls_token = nn.Parameter(torch.zeros(n_dim))
        self.add_residual = add_residual
        print(f'Add residual {add_residual}')

        if use_t_conv:
            self.tconv = nn.ModuleList([
                nn.Conv1d(n_dim, n_dim, kernel_size=3, stride=1, padding=1, bias=True, groups=n_dim)
                for i in range(n_layers)
            ])
            for m in self.tconv:
                nn.init.constant_(m.bias, 0.)
                m.weight.data[...] = torch.Tensor([0, 1, 0])
        else:
            self.tconv = None

        if use_t_pos_embed:
            self.pemb_t = nn.Parameter(torch.zeros([n_layers, t_size, n_dim]))
        else:
            self.pemb_t = None

        self.t_size = t_size

    def forward(self, clip_feats_all):
        # clip_feats_all = clip_feats_all[-len(self.dec):]
        # only return n_layers features, save memory
        clip_feats = [x for x in clip_feats_all]

        L, N, T, C = clip_feats[0].size()
        x = self.temporal_cls_token.view(1, 1, -1).repeat(1, N, 1)

        for i in range(len(clip_feats)):
            if self.tconv is not None:
                L, N, T, C = clip_feats[i].shape
                clip_feats[i] = clip_feats[i].permute(0, 1, 3, 2).flatten(0, 1)  # L * N, C, T
                clip_feats[i] = self.tconv[i](clip_feats[i]).permute(0, 2, 1).contiguous().view(L, N, T, C)
            if self.pemb_t is not None:
                clip_feats[i] = clip_feats[i] + self.pemb_t[i]
            clip_feats[i] = clip_feats[i].permute(2, 0, 1, 3).flatten(0, 1)  # T * L, N, C

        for i in range(len(self.dec)):
            x = self.dec[i](x, clip_feats[i])

        if self.add_residual:
            residual = clip_feats_all[-1][0].mean(1)
            return self.proj(x[0, :, :] + residual)
        else:
            return self.proj(x[0, :, :])


if __name__ == '__main__':
    model = TransformerDecoder()

    # construct a fake input to demonstrate input tensor shape
    L, N, T, C = 197, 1, 8, 768  # num_image_tokens, video_batch_size, t_size, feature_dim
    # we use intermediate feature maps from multiple blocks, so input features should be a list
    input_features = []
    for i in range(4):  # vit-b has 12 blocks
        # every item in input_features contains features maps from a single block
        # every item is a tuple containing 3 feature maps:
        # (1) block output features (i.e. after mlp) with shape L, N, T, C
        # (2) projected query features with shape L, N, T, C
        # (3) projected key features with shape L, N, T, C
        input_features.append(
            torch.zeros([L, N, T, C]))
        # some small optimizations:
        # (1) We only decode from the last $n$ blocks so it's good as long as the last $n$ items of input_features is valid and all previous items can be filled with None to save memory. By default $n=4$.
        # (2) projected query/key features are optional. If you are using an uncompatible image backbone without query/key (e.g. CNN), you can fill the position with None (i.e. the tuple should be (Tensor, None, None) and set use_image_attnmap=False when constructing the model.

    print(model(input_features).shape)  # should be N, 400
