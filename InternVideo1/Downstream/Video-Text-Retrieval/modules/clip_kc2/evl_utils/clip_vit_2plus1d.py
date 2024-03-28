import os
from collections import OrderedDict

from timm.models.layers import DropPath
import torch
from torch import nn
from einops import rearrange

from .attention import MultiheadAttention


MODEL_PATH = '/mnt/lustre/share_data/likunchang.vendor/model'
_MODELS = {
    "ViT-B/32": os.path.join(MODEL_PATH, "vit_b32.pth"),
    "ViT-B/16": os.path.join(MODEL_PATH, "vit_b16.pth"),
    "ViT-L/14": os.path.join(MODEL_PATH, "vit_l14.pth"),
    "ViT-L/14_336": os.path.join(MODEL_PATH, "vit_l14_336.pth"),
}


class LayerNorm(nn.LayerNorm):
    """Subclass torch's LayerNorm to handle fp16."""

    def forward(self, x):
        orig_type = x.dtype
        ret = super().forward(x.type(torch.float32))
        return ret.type(orig_type)


class QuickGELU(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(1.702 * x)


class ResidualAttentionBlock(nn.Module):
    def __init__(self, d_model, n_head, attn_mask=None, drop_path=0.0):
        super().__init__()
        
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        print(f'Drop path rate: {drop_path}')
        # temporal
        self.attn_t = MultiheadAttention(d_model, n_head)
        self.ln_t = LayerNorm(d_model)
        # spatial
        self.attn = MultiheadAttention(d_model, n_head)
        self.ln_1 = LayerNorm(d_model)
        self.mlp = nn.Sequential(OrderedDict([
            ("c_fc", nn.Linear(d_model, d_model * 4)),
            ("gelu", QuickGELU()),
            ("c_proj", nn.Linear(d_model * 4, d_model))
        ]))
        self.ln_2 = LayerNorm(d_model)
        self.attn_mask = attn_mask
        
        # init zero
        print('Init zero for (2+1)d')
        nn.init.constant_(self.attn_t.in_proj_weight, 0)
        nn.init.constant_(self.attn_t.in_proj_bias, 0)
        nn.init.constant_(self.attn_t.out_proj.weight, 1)
        nn.init.constant_(self.attn_t.out_proj.bias, 0)

    def attention(self, x):
        self.attn_mask = self.attn_mask.to(dtype=x.dtype, device=x.device) if self.attn_mask is not None else None
        return self.attn(x, x, x, need_weights=False, attn_mask=self.attn_mask)[0]

    def attention_temporal(self, x):
        self.attn_mask = None
        return self.attn_t(x, x, x, need_weights=False, attn_mask=self.attn_mask)[0]

    def forward(self, x, T=8):
        # temporal
        # x: 1+HWT, N, C
        xt = x[1:, :, :]
        _, N, C = xt.shape
        xt = rearrange(xt, '(l t) n c -> t (n l) c', n=N, t=T)
        res_temporal = self.attention_temporal(self.ln_t(xt))
        res_temporal = rearrange(res_temporal, 't (n l) c -> (l t) n c', n=N, t=T)
        xt = x[1:, :, :] + self.drop_path(res_temporal)

        # spatial
        init_cls_token = x[:1, :, :]
        cls_token = init_cls_token.repeat(1, T, 1).view(1, T*N, C)
        xs = rearrange(xt, '(l t) n c -> l (t n) c', n=N, t=T)
        xs = torch.cat((cls_token, xs), 0)
        res_spatial = self.attention(self.ln_1(xs))

        # Taking care of CLS token
        cls_token = res_spatial[0, :, :]
        cls_token = rearrange(cls_token, '(t n) c -> t n c', n=N)
        cls_token = torch.mean(cls_token, 0, True) # averaging for every frame
        res_spatial = res_spatial[1:, :, :]
        res_spatial = rearrange(res_spatial, 'l (t n) c -> (l t) n c', n=N)
        x = x + self.drop_path(torch.cat((cls_token, res_spatial), 0))

        x = x + self.drop_path(self.mlp(self.ln_2(x)))
        return x


class Transformer(nn.Module):
    def __init__(self, width, layers, heads, attn_mask=None, drop_path_rate=0.):
        super().__init__()
        self.width = width
        self.layers = layers
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, layers)]
        self.resblocks = nn.ModuleList([
            ResidualAttentionBlock(width, heads, attn_mask, drop_path=dpr[i]) for i in range(layers)
        ])

    def forward(self, x, return_num=4, T=8):
        features = []
        for i, resblock in enumerate(self.resblocks):
            x = resblock(x, T=T)
            if i >= self.layers - return_num:
                # LT + 1, N, C
                LT, N, C = x.shape
                L = (LT - 1) // T
                cls_x, tmp_x = x[:1], x[1:]
                cls_x = cls_x.unsqueeze(2).repeat(1, 1, T, 1)
                tmp_x = tmp_x.reshape(L, T, N, C).permute(0, 2, 1, 3) # L, N, T, C
                tmp_x = torch.cat([cls_x, tmp_x], dim=0 )# L + 1, N, T, C
                features.append(tmp_x)
        return features


class VisionTransformer(nn.Module):
    def __init__(
        self, input_resolution, patch_size, width, layers, heads, output_dim, num_frames=8, drop_path_rate=0.,
    ):
        super().__init__()
        self.input_resolution = input_resolution
        self.output_dim = output_dim
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=width, kernel_size=patch_size, stride=patch_size, bias=False)

        scale = width ** -0.5
        self.class_embedding = nn.Parameter(scale * torch.randn(width))
        self.positional_embedding = nn.Parameter(scale * torch.randn((input_resolution // patch_size) ** 2 + 1, width))
        self.temporal_positional_embedding = nn.Parameter(torch.zeros(1, num_frames, width))
        self.ln_pre = LayerNorm(width)

        self.transformer = Transformer(width, layers, heads, drop_path_rate=drop_path_rate)

  

    def forward(self, x, return_num=4):
        N, C, T, H, W = x.shape
        x = x.permute(0, 2, 1, 3, 4).reshape(N * T, C, H, W)

        x = self.conv1(x)  # shape = [*, width, grid, grid]
        x = x.reshape(x.shape[0], x.shape[1], -1)  # shape = [*, width, grid ** 2]
        x = x.permute(0, 2, 1)  # shape = [*, grid ** 2, width]
        x = torch.cat([self.class_embedding.to(x.dtype) + torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device), x], dim=1)  # shape = [*, grid ** 2 + 1, width]
        x = x + self.positional_embedding.to(x.dtype)

        cls_tokens = x[:N, :1, :]
        x = x[:, 1:]
        x = rearrange(x, '(b t) n c -> (b n) t c', b=N, t=T)
        # x = x + self.temporal_positional_embedding
        x = x + self.temporal_positional_embedding
        x = rearrange(x, '(b n) t c -> b (n t) c', b=N, t=T)
        x = torch.cat((cls_tokens, x), dim=1)

        x = self.ln_pre(x)

        x = x.permute(1, 0, 2)  # NLD -> LND
        features = self.transformer(
            x, return_num=return_num, T=T,
        )

        return features


def vit_2plus1d_b32(pretrained=True, num_frames=8, drop_path_rate=0.):
    model = VisionTransformer(
        input_resolution=224,
        patch_size=32,
        width=768,
        layers=12,
        heads=12,
        output_dim=512,
        num_frames=num_frames,
        drop_path_rate=drop_path_rate
    )

    if pretrained:
        print('load pretrained weights')
        state_dict = torch.load(_MODELS["ViT-B/32"], map_location='cpu')
        model.load_state_dict(state_dict, strict=False)
    return model.eval()


def vit_2plus1d_b16(pretrained=True, num_frames=8, drop_path_rate=0.):
    model = VisionTransformer(
        input_resolution=224,
        patch_size=16,
        width=768,
        layers=12,
        heads=12,
        output_dim=512,
        num_frames=num_frames,
        drop_path_rate=drop_path_rate,
    )

    if pretrained:
        print('load pretrained weights')
        state_dict = torch.load(_MODELS["ViT-B/16"], map_location='cpu')
        model.load_state_dict(state_dict, strict=False)
    return model.eval()


def vit_2plus1d_l14(pretrained=True, num_frames=8, drop_path_rate=0.):
    model = VisionTransformer(
        input_resolution=224,
        patch_size=14,
        width=1024,
        layers=24,
        heads=16,
        output_dim=768,
        num_frames=num_frames,
        drop_path_rate=drop_path_rate
    )

    if pretrained:
        print('load pretrained weights')
        state_dict = torch.load(_MODELS["ViT-L/14"], map_location='cpu')
        model.load_state_dict(state_dict, strict=False)
    return model.eval()


def vit_2plus1d_l14_336(pretrained=True, num_frames=8, drop_path_rate=0.):
    model = VisionTransformer(
        input_resolution=336,
        patch_size=14,
        width=1024,
        layers=24,
        heads=16,
        output_dim=768,
        num_frames=num_frames,
        drop_path_rate=drop_path_rate
    )

    if pretrained:
        print('load pretrained weights')
        state_dict = torch.load(_MODELS["ViT-L/14_336"], map_location='cpu')
        model.load_state_dict(state_dict, strict=False)
    return model.eval()


if __name__ == '__main__':
    import time
    from fvcore.nn import FlopCountAnalysis
    from fvcore.nn import flop_count_table

    model = vit_2plus1d_b32(pretrained=True)

    flops = FlopCountAnalysis(model, torch.rand(1, 3, 8, 224, 224))
    s = time.time()
    print(flop_count_table(flops, max_depth=1))
    print(time.time()-s)