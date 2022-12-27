import os
from collections import OrderedDict

from timm.models.layers import DropPath
import torch
from torch import nn
from einops import rearrange
import torch.utils.checkpoint as checkpoint

from .attention_bias import MultiheadAttention
from ipdb import set_trace

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
    def __init__(self, d_model, n_head, attn_mask=None, drop_path=0.0, t_size=8, spatial_size=7, init_zero=True):
        super().__init__() 
        
        self.n_head = n_head
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        print(f'Drop path rate: {drop_path}')
        print(f'Add RPB: t_size {t_size}, spatial_size {spatial_size}')
        self.pos_embed = nn.Conv3d(d_model, d_model, kernel_size=3, stride=1, padding=1, groups=d_model)
        # temporal
        self.attn_t = MultiheadAttention(d_model, n_head)
        self.ln_t = LayerNorm(d_model)
        self.rpb_t = nn.Parameter(torch.zeros([t_size * 2 - 1, n_head]))

        idx_tensor_t = torch.zeros([t_size, t_size], dtype=torch.long)
        for q in range(t_size):
            for k in range(t_size):
                offs = q - k + t_size - 1
                idx_tensor_t[q, k] = offs
        self.idx_tensor_t = idx_tensor_t

        # spatial
        self.attn = MultiheadAttention(d_model, n_head)
        self.rpb = nn.Parameter(torch.zeros([(spatial_size * 2 - 1) ** 2, n_head]))
        self.ln_1 = LayerNorm(d_model)
        self.mlp = nn.Sequential(OrderedDict([
            ("c_fc", nn.Linear(d_model, d_model * 4)),
            ("gelu", QuickGELU()),
            ("c_proj", nn.Linear(d_model * 4, d_model))
        ]))
        self.ln_2 = LayerNorm(d_model)
        self.attn_mask = attn_mask

        idx_tensor = torch.zeros([spatial_size ** 2, spatial_size ** 2], dtype=torch.long)
        for q in range(spatial_size ** 2):
            qi, qj = q // spatial_size, q % spatial_size
            for k in range(spatial_size ** 2):
                ki, kj = k // spatial_size, k % spatial_size
                i_offs = qi - ki + spatial_size - 1
                j_offs = qj - kj + spatial_size - 1
                idx_tensor[q, k] = i_offs * (spatial_size * 2 - 1) + j_offs
        self.idx_tensor = idx_tensor
        
        if init_zero:
            # init zero
            print('Init zero for (2+1)d')
            nn.init.constant_(self.pos_embed.weight, 0)
            nn.init.constant_(self.pos_embed.bias, 0)
            nn.init.constant_(self.attn_t.in_proj_weight, 0)
            nn.init.constant_(self.attn_t.in_proj_bias, 0)
            nn.init.constant_(self.attn_t.out_proj.weight, 1)
            nn.init.constant_(self.attn_t.out_proj.bias, 0)
            nn.init.constant_(self.ln_t.weight, 1.)
            nn.init.constant_(self.ln_t.bias, 0.)
        else:
            nn.init.trunc_normal_(self.rpb_t, std=.02)
            nn.init.trunc_normal_(self.rpb, std=.02)

            nn.init.trunc_normal_(self.pos_embed.weight, std=.02)
            nn.init.constant_(self.pos_embed.bias, 0)
            nn.init.trunc_normal_(self.attn_t.in_proj_weight, std=.02)
            nn.init.constant_(self.attn_t.in_proj_bias, 0)
            nn.init.trunc_normal_(self.attn_t.out_proj.weight, std=.02)
            nn.init.constant_(self.attn_t.out_proj.bias, 0)
            nn.init.constant_(self.ln_t.weight, 1.)
            nn.init.constant_(self.ln_t.bias, 0.)

    def attention(self, x, rpb=None):
        self.attn_mask = self.attn_mask.to(dtype=x.dtype, device=x.device) if self.attn_mask is not None else None
        return self.attn(x, x, x, need_weights=False, attn_mask=self.attn_mask, rpb=rpb)[0]

    def attention_temporal(self, x, rpb=None):
        self.attn_mask = None
        return self.attn_t(x, x, x, need_weights=False, attn_mask=self.attn_mask, rpb=rpb)[0]

    def forward(self, x, T=8, mode='video'):
        # temporal
        # x: 1+HWT, N, C
        # pos_emb
        tmp_x = x[1:, :, :]
        LT, N, C = tmp_x.shape
        L = LT // T
        H = W = int(L ** 0.5)
        tmp_x = tmp_x.view(H, W, T, N, C).permute(3, 4, 2, 0, 1)
        tmp_x = tmp_x + self.pos_embed(tmp_x)
        tmp_x = tmp_x.view(N, C, T, L).permute(3, 2, 0, 1).view(LT, N, C)
        x[1:, :, :] = tmp_x

        xt = x[1:, :, :]
        _, N, C = xt.shape
        xt = rearrange(xt, '(l t) n c -> t (n l) c', n=N, t=T)
        # no rpb_t for image
        if mode == 'image':
            rpb_t = None
        else:
            # rpb_t: T, T, H => B*H, T, T
            self.idx_tensor_t = self.idx_tensor_t.to(xt.device)
            rpb_t = self.rpb_t[self.idx_tensor_t].permute(2, 0, 1).repeat(N*L, 1, 1)
        
        # set_trace()
        res_temporal = self.attention_temporal(self.ln_t(xt), rpb=rpb_t)
        res_temporal = rearrange(res_temporal, 't (n l) c -> (l t) n c', n=N, t=T)
        xt = x[1:, :, :] + self.drop_path(res_temporal)

        # spatial
        init_cls_token = x[:1, :, :]
        cls_token = init_cls_token.repeat(1, T, 1).view(1, T*N, C)
        xs = rearrange(xt, '(l t) n c -> l (t n) c', n=N, t=T)
        xs = torch.cat((cls_token, xs), 0)
        # rpb: L, L, H => B*H, L+1, L+1
        rpb = torch.zeros((self.n_head, L+1, L+1), device=xs.device, dtype=xs.dtype)
        self.idx_tensor = self.idx_tensor.to(xs.device)
        rpb[:, 1:, 1:] = self.rpb[self.idx_tensor].permute(2, 0, 1)
        rpb = rpb.repeat(T*N, 1, 1)
        res_spatial = self.attention(self.ln_1(xs), rpb=rpb)

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
    def __init__(self, width, layers, heads, attn_mask=None, drop_path_rate=0., t_size=8, spatial_size=7, init_zero=True):
        super().__init__()
        self.width = width
        self.layers = layers
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, layers)]
        self.resblocks = nn.ModuleList([
            ResidualAttentionBlock(width, heads, attn_mask, 
            drop_path=dpr[i], t_size=t_size, spatial_size=spatial_size, init_zero=init_zero) for i in range(layers)
        ])

    def forward(self, x, return_num=4, T=8, mode='video'):
        features = []
        for i, resblock in enumerate(self.resblocks):
            x = resblock(x, T=T, mode=mode)
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
        self, input_resolution, patch_size, width, layers, heads, output_dim, 
        num_frames=8, drop_path_rate=0., t_size=8, spatial_size=7, init_zero=True,
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

        self.transformer = Transformer(width, layers, heads, drop_path_rate=drop_path_rate, t_size=t_size, spatial_size=spatial_size, init_zero=init_zero)
        self.mask_embedding = nn.Parameter(scale * torch.randn(width))
        
        print('-' * 100)
        print('tsize:', t_size, 'num frame: ', num_frames)
        print('-' * 100)

    def forward(self, x, return_num=4, masked_indices=None, mode='video'):
        if len(x.size()) == 5:
            N, C, T, H, W = x.shape
            x = x.permute(0, 2, 1, 3, 4).reshape(N * T, C, H, W)

        x = self.conv1(x)  # shape = [*, width, grid, grid]
        x = x.reshape(x.shape[0], x.shape[1], -1)  # shape = [*, width, grid ** 2]
        x = x.permute(0, 2, 1)  # shape = [*, grid ** 2, width]
        if masked_indices is not None:
            masked_indices = masked_indices.view(N * T, -1)
            x[masked_indices] = self.mask_embedding.type(x.dtype)
        x = torch.cat([self.class_embedding.to(x.dtype) + torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device), x], dim=1)  # shape = [*, grid ** 2 + 1, width]
        x = x + self.positional_embedding.to(x.dtype)

        cls_tokens = x[:N, :1, :]
        x = x[:, 1:]
        # add temporal position embedding for video
        if mode == 'video':
            x = rearrange(x, '(b t) n c -> (b n) t c', b=N, t=T)
            # x = x + self.temporal_positional_embedding
            x = x + self.temporal_positional_embedding
            x = rearrange(x, '(b n) t c -> b (n t) c', b=N, t=T)
        else:
            pass
        x = torch.cat((cls_tokens, x), dim=1)

        x = self.ln_pre(x)

        x = x.permute(1, 0, 2)  # NLD -> LND
        features = self.transformer(
            x, return_num=return_num, T=T, mode=mode
        )

        return features


def vit_2plus1d_dw_bias_b32(pretrained=True, num_frames=8, drop_path_rate=0., t_size=8, init_zero=True):
    model = VisionTransformer(
        input_resolution=224,
        patch_size=32,
        width=768,
        layers=12,
        heads=12,
        output_dim=512,
        num_frames=num_frames,
        drop_path_rate=drop_path_rate,
        init_zero=init_zero,
        t_size=t_size,
    )

    if pretrained:
        print('load pretrained weights')
        state_dict = torch.load(_MODELS["ViT-B/32"], map_location='cpu')
        model.load_state_dict(state_dict, strict=False)
    return model.eval()


def vit_2plus1d_dw_bias_b16(pretrained=True, num_frames=8, drop_path_rate=0., t_size=8, init_zero=True):
    model = VisionTransformer(
        input_resolution=224,
        patch_size=16,
        width=768,
        layers=12,
        heads=12,
        output_dim=512,
        num_frames=num_frames,
        drop_path_rate=drop_path_rate,
        t_size=t_size,
        spatial_size=14,
        init_zero=init_zero,
    )

    if pretrained:
        print('load pretrained weights')
        state_dict = torch.load(_MODELS["ViT-B/16"], map_location='cpu')
        model.load_state_dict(state_dict, strict=False)
    return model.eval()


def vit_2plus1d_dw_bias_l14(pretrained=True, num_frames=8, drop_path_rate=0., t_size=8, init_zero=True):
    model = VisionTransformer(
        input_resolution=224,
        patch_size=14,
        width=1024,
        layers=24,
        heads=16,
        output_dim=768,
        num_frames=num_frames,
        drop_path_rate=drop_path_rate,
        t_size=t_size,
        init_zero=init_zero,
    )

    if pretrained:
        print('load pretrained weights')
        state_dict = torch.load(_MODELS["ViT-L/14"], map_location='cpu')
        model.load_state_dict(state_dict, strict=False)
    return model.eval()


def vit_2plus1d_dw_bias_l14_336(pretrained=True, num_frames=8, drop_path_rate=0., t_size=8, init_zero=True):
    model = VisionTransformer(
        input_resolution=336,
        patch_size=14,
        width=1024,
        layers=24,
        heads=16,
        output_dim=768,
        num_frames=num_frames,
        drop_path_rate=drop_path_rate,
        t_size=t_size,
        init_zero=init_zero,
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

    model = vit_2plus1d_dw_bias_b32(pretrained=True)

    flops = FlopCountAnalysis(model, torch.rand(4, 3, 8, 224, 224))
    s = time.time()
    print(flop_count_table(flops, max_depth=1))
    print(time.time()-s)