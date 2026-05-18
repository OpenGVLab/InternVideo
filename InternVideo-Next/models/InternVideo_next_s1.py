# Code Refactored by Claude Code.

import math
import torch
import torch.nn.functional as F
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from timm.models.registry import register_model
from torch import nn

import torch.utils.checkpoint as checkpoint
from functools import partial
import einops
from einops import rearrange

from .pos_embed import get_3d_sincos_pos_embed, get_2d_sincos_pos_embed, get_1d_sincos_pos_embed
from .flash_attention_class import FlashAttention
from .diffloss import DiffLoss
from flash_attn.modules.mlp import FusedMLP
from flash_attn.ops.rms_norm import DropoutAddRMSNorm

def mask_with_cls(mask_in, xinput, with_cls=True, cls_token_num=None):
    _B, _C = xinput.shape[0], xinput.shape[-1]

    if len(xinput.shape) == 4:
        xinput = xinput.reshape(_B, -1, _C)

    if mask_in is not None:
        if with_cls:
            cls_tokens, xinput = xinput[:, :cls_token_num, :].reshape(_B, -1, _C), xinput[:, cls_token_num:, :].reshape(_B, -1, _C)
        if isinstance(mask_in, list):
            for m in mask_in:
                xinput = xinput[~m].reshape(_B, -1, _C)
        else:
            xinput = xinput[~mask_in].reshape(_B, -1, _C)

        if with_cls:
            xinput = torch.cat((cls_tokens, xinput), dim=1)

    return xinput


class CrossAttention(nn.Module):
    def __init__(
            self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0.,
            proj_drop=0., attn_head_dim=None, out_dim=None):
        super().__init__()
        if out_dim is None:
            out_dim = dim
        self.num_heads = num_heads
        head_dim = dim // num_heads
        if attn_head_dim is not None:
            head_dim = attn_head_dim
        all_head_dim = head_dim * self.num_heads
        self.scale = qk_scale or head_dim ** -0.5
        assert all_head_dim == dim

        self.q = nn.Linear(dim, all_head_dim, bias=False)
        self.k = nn.Linear(dim, all_head_dim, bias=False)
        self.v = nn.Linear(dim, all_head_dim, bias=False)

        if qkv_bias:
            self.q_bias = nn.Parameter(torch.zeros(all_head_dim))
            self.k_bias = nn.Parameter(torch.zeros(all_head_dim))
            self.v_bias = nn.Parameter(torch.zeros(all_head_dim))
        else:
            self.q_bias = None
            self.k_bias = None
            self.v_bias = None

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(all_head_dim, out_dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x, k=None, v=None):
        B, N, C = x.shape
        N_k = k.shape[1]
        N_v = v.shape[1]

        q_bias, k_bias, v_bias = None, None, None
        if self.q_bias is not None:
            q_bias = self.q_bias
            k_bias = self.k_bias
            v_bias = self.v_bias

        q = F.linear(input=x, weight=self.q.weight, bias=q_bias)
        q = q.reshape(B, N, 1, self.num_heads, -1).permute(2, 0, 3, 1, 4).squeeze(0)

        k = F.linear(input=k, weight=self.k.weight, bias=k_bias)
        k = k.reshape(B, N_k, 1, self.num_heads, -1).permute(2, 0, 3, 1, 4).squeeze(0)

        v = F.linear(input=v, weight=self.v.weight, bias=v_bias)
        v = v.reshape(B, N_v, 1, self.num_heads, -1).permute(2, 0, 3, 1, 4).squeeze(0)

        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))

        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, -1)
        x = self.proj(x)
        x = self.proj_drop(x)

        return x


class AttentiveBlock(nn.Module):

    def __init__(self, dim, num_heads, qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., norm_layer=nn.LayerNorm, attn_head_dim=None, out_dim=None):
        super().__init__()

        self.norm1_q = norm_layer(dim)
        self.norm1_k = norm_layer(dim)
        self.norm1_v = norm_layer(dim)
        self.cross_attn = CrossAttention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop,
            proj_drop=drop, attn_head_dim=attn_head_dim, out_dim=out_dim)

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x_q, x_kv, pos_q, pos_k, bool_masked_pos, rel_pos_bias=None):
        x_q = self.norm1_q(x_q + pos_q)
        x_k = self.norm1_k(x_kv + pos_k)
        x_v = self.norm1_v(x_kv)
        x = self.cross_attn(x_q, k=x_k, v=x_v)

        return x


class AttentionPoolingBlock(AttentiveBlock):

    def forward(self, x):
        x_q = x.mean(1, keepdim=True)
        x_kv, pos_q, pos_k = x, 0, 0
        x = super().forward(x_q, x_kv, pos_q, pos_k, bool_masked_pos=None, rel_pos_bias=None)
        x = x.squeeze(1)
        return x


class RMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)


class LayerScale(nn.Module):
    def __init__(self, dim, init_values=1e-5, inplace=False, force_fp32=False):
        super().__init__()
        self.inplace = inplace
        self.gamma = nn.Parameter(init_values * torch.ones(dim))
        self.force_fp32 = force_fp32

    @torch.cuda.amp.autocast(enabled=False)
    def forward(self, x):
        if self.force_fp32:
            output_type = x.dtype
            out = x.float().mul_(self.gamma.float()) if self.inplace else x.float() * self.gamma.float()
            return out.to(dtype=output_type)
        else:
            out = x.mul_(self.gamma) if self.inplace else x * self.gamma
            return out


def rotate_queries_or_keys(x, pos):
    x = x.permute(0, 2, 1, 3)
    B, num_head, N, D = x.size()
    assert D % 2 == 0, "Embedding dimension must be a multiple of 2 for block matrix rotation"

    # -- compute angle for each position
    omega = torch.arange(D // 2, dtype=x.dtype, device=x.device)
    omega /= D / 2.0
    omega = 1.0 / 10000**omega  # (D/2,)
    freq = torch.einsum("..., f -> ... f", pos, omega)  # (..., N, D/2), outer product

    # -- build rotation matrix and apply
    emb_sin = freq.sin()  # (..., N, D/2)
    emb_cos = freq.cos()  # (..., N, D/2)

    emb_sin = emb_sin.squeeze(-1).repeat(1, 1, 2).unsqueeze(1)
    emb_cos = emb_cos.squeeze(-1).repeat(1, 1, 2).unsqueeze(1)

    # --
    y = x.unflatten(-1, (-1, 2))
    y1, y2 = y.unbind(dim=-1)
    y = torch.stack((-y2, y1), dim=-1)
    y = y.flatten(-2)

    return ((x * emb_cos) + (y * emb_sin)).permute(0, 2, 1, 3)


class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0., use_flash_attn=False,
                 causal=False, norm_layer=nn.LayerNorm, qk_normalization=False, use_fused_rmsnorm=False,
                 use_rope=False, cls_token_num=4):
        super().__init__()
        assert dim % num_heads == 0, 'dim should be divisible by num_heads'
        self.num_heads = num_heads
        self.cls_token_num = cls_token_num
        head_dim = dim // num_heads
        self.head_dim = head_dim
        self.scale = head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.use_flash_attn = use_flash_attn

        if use_flash_attn:
            self.causal = causal
            self.inner_attn = FlashAttention(attention_dropout=attn_drop)

        self.use_rope = use_rope
        if use_rope:
            self.d_dim = int(2 * ((head_dim // 3) // 2))
            self.h_dim = int(2 * ((head_dim // 3) // 2))
            self.w_dim = int(2 * ((head_dim // 3) // 2))

        self.qk_normalization = qk_normalization
        self.q_norm = norm_layer(dim) if qk_normalization else nn.Identity()
        self.k_norm = norm_layer(dim) if qk_normalization else nn.Identity()
        self.use_fused_rmsnorm = use_fused_rmsnorm

    def _naive_attn(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)

        if self.qk_normalization:
            B_, H_, N_, D_ = q.shape
            q = self.q_norm(q.transpose(1, 2).flatten(-2, -1)).view(B_, N_, H_, D_).transpose(1, 2)
            k = self.k_norm(k.transpose(1, 2).flatten(-2, -1)).view(B_, N_, H_, D_).transpose(1, 2)

        attn = ((q * self.scale) @ k.transpose(-2, -1))
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

    def _flash_attn(self, x, key_padding_mask=None, need_weights=False):

        qkv = self.qkv(x)
        qkv = rearrange(qkv, "b s (three h d) -> b s three h d", three=3, h=self.num_heads)

        if self.qk_normalization:
            q, k, v = qkv.unbind(2)
            if self.use_fused_rmsnorm:
                q = self.q_norm(q.flatten(-2, -1))[0].view(q.shape)
                k = self.k_norm(k.flatten(-2, -1))[0].view(k.shape)
            else:
                q = self.q_norm(q.flatten(-2, -1)).view(q.shape)
                k = self.k_norm(k.flatten(-2, -1)).view(k.shape)
            qkv = torch.stack([q, k, v], dim=2)

        context, _ = self.inner_attn(
            qkv, key_padding_mask=key_padding_mask, need_weights=need_weights, causal=self.causal
        )
        outs = self.proj(rearrange(context, "b s h d -> b s (h d)"))
        outs = self.proj_drop(outs)
        return outs

    def _get_frame_pos(self, ids):
        tokens_per_frame = int(14 * 14)
        return ids // tokens_per_frame

    def _get_height_pos(self, ids):
        tokens_per_frame = int(14 * 14)
        tokens_per_row = 14
        frame_ids = self._get_frame_pos(ids)
        ids = ids - tokens_per_frame * frame_ids
        return ids // tokens_per_row

    def separate_positions(self, ids):
        tokens_per_frame = int(14 * 14)
        tokens_per_row = 14
        frame_ids = self._get_frame_pos(ids)
        height_ids = self._get_height_pos(ids)
        width_ids = (ids - tokens_per_frame * frame_ids) - tokens_per_row * height_ids
        return frame_ids, height_ids, width_ids

    def _flash_attn_w_rope(self, x, mask, key_padding_mask=None, need_weights=False):
        B, N, C = x.shape

        qkv = self.qkv(x)
        qkv = rearrange(qkv, "b s (three h d) -> b s three h d", three=3, h=self.num_heads)

        qkv_cls, qkv = qkv[:, :self.cls_token_num, :, :], qkv[:, self.cls_token_num:, :, :]

        q, k, v = qkv.unbind(2)

        def mask_to_ids(mask_in):
            mask_in = mask_in.squeeze(-1)
            mask_ids_n = torch.where(mask_in == 0)[1]
            return mask_ids_n.reshape(B, -1).to(mask_in.device)

        unmasked_ids = mask_to_ids(mask)

        d_mask, h_mask, w_mask = self.separate_positions(unmasked_ids)

        s = 0
        qd = rotate_queries_or_keys(q[..., s:s + self.d_dim], pos=d_mask)
        kd = rotate_queries_or_keys(k[..., s:s + self.d_dim], pos=d_mask)
        s += self.d_dim
        qh = rotate_queries_or_keys(q[..., s:s + self.h_dim], pos=h_mask)
        kh = rotate_queries_or_keys(k[..., s:s + self.h_dim], pos=h_mask)
        s += self.h_dim
        qw = rotate_queries_or_keys(q[..., s:s + self.w_dim], pos=w_mask)
        kw = rotate_queries_or_keys(k[..., s:s + self.w_dim], pos=w_mask)
        s += self.w_dim

        if s < self.head_dim:
            qr = q[..., s:]
            kr = k[..., s:]
            q = torch.cat([qd, qh, qw, qr], dim=-1)
            k = torch.cat([kd, kh, kw, kr], dim=-1)
        else:
            q = torch.cat([qd, qh, qw], dim=-1)
            k = torch.cat([kd, kh, kw], dim=-1)

        qkv = torch.stack((q, k, v), dim=2)
        qkv = torch.cat((qkv_cls, qkv), dim=1)

        if self.qk_normalization:
            q, k, v = qkv.unbind(2)
            if self.use_fused_rmsnorm:
                q = self.q_norm(q.flatten(-2, -1))[0].view(q.shape)
                k = self.k_norm(k.flatten(-2, -1))[0].view(k.shape)
            else:
                q = self.q_norm(q.flatten(-2, -1)).view(q.shape)
                k = self.k_norm(k.flatten(-2, -1)).view(k.shape)
            qkv = torch.stack([q, k, v], dim=2)

        context, _ = self.inner_attn(
            qkv, key_padding_mask=key_padding_mask, need_weights=need_weights, causal=self.causal
        )
        outs = self.proj(rearrange(context, "b s h d -> b s (h d)"))
        outs = self.proj_drop(outs)
        return outs

    def forward(self, x, mask):
        x = self._naive_attn(x) if not self.use_flash_attn else (
            self._flash_attn(x) if not self.use_rope else self._flash_attn_w_rope(x, mask))
        return x


class Mlp(nn.Module):
    """MLP as used in Vision Transformer, MLP-Mixer and related networks."""

    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU,
                 bias=True, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        bias = to_2tuple(bias)
        drop_probs = to_2tuple(drop)

        self.fc1 = nn.Linear(in_features, hidden_features, bias=bias[0])
        self.act = act_layer()
        self.drop1 = nn.Dropout(drop_probs[0])
        self.fc2 = nn.Linear(hidden_features, out_features, bias=bias[1])
        self.drop2 = nn.Dropout(drop_probs[1])

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop1(x)
        x = self.fc2(x)
        x = self.drop2(x)
        return x


class Block(nn.Module):

    def __init__(
            self, dim, num_heads, mlp_ratio=4., qkv_bias=False, drop=0., attn_drop=0., init_values=None,
            drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, use_flash_attn=False, use_fused_mlp=False,
            fused_mlp_heuristic=1, with_cp=False, qk_normalization=False, layerscale_no_force_fp32=False,
            use_rope=False, use_fused_rmsnorm=False, cls_token_num=4):
        super().__init__()

        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop,
            use_flash_attn=use_flash_attn, causal=False, norm_layer=norm_layer,
            qk_normalization=qk_normalization, use_fused_rmsnorm=use_fused_rmsnorm, use_rope=use_rope,
            cls_token_num=cls_token_num)
        self.ls1 = LayerScale(dim, init_values=init_values,
                               force_fp32=(not layerscale_no_force_fp32)) if init_values else nn.Identity()
        self.drop_path1 = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        if use_fused_mlp:
            self.mlp = FusedMLP(in_features=dim, hidden_features=mlp_hidden_dim, heuristic=fused_mlp_heuristic)
        else:
            self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)
        self.ls2 = LayerScale(dim, init_values=init_values,
                               force_fp32=(not layerscale_no_force_fp32)) if init_values else nn.Identity()
        self.drop_path2 = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        self.with_cp = with_cp
        self.use_fused_rmsnorm = use_fused_rmsnorm

    def forward(self, x, residual=None, mask=None):

        def _inner_forward(x, residual=None):
            if self.use_fused_rmsnorm:
                x, residual = self.norm1(x, residual)
                x = self.drop_path1(self.ls1(self.attn(x, mask=mask)))
                x, residual = self.norm2(x, residual)
                x = self.drop_path2(self.ls2(self.mlp(x)))
                return x, residual
            else:
                assert residual is None
                x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x), mask=mask)))
                x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
                return x

        if self.with_cp:
            return checkpoint.checkpoint(_inner_forward, x, residual)
        else:
            return _inner_forward(x, residual=residual)


class PatchEmbed(nn.Module):
    """3D Image to Patch Embedding."""

    def __init__(
            self, img_size=224, patch_size=16, in_chans=3, embed_dim=768,
            num_frames=8, tubelet_size=1, norm_layer=None, dual_norm_in_patch_embed=True):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        self.img_size = img_size
        self.patch_size = patch_size
        self.tubelet_size = tubelet_size
        self.grid_size = (
            num_frames // tubelet_size,
            img_size[0] // patch_size[0],
            img_size[1] // patch_size[1]
        )
        self.num_patches = self.grid_size[0] * self.grid_size[1] * self.grid_size[2]

        self.proj = nn.Conv3d(
            in_channels=in_chans, out_channels=embed_dim,
            kernel_size=(tubelet_size, patch_size[0], patch_size[1]),
            stride=(tubelet_size, patch_size[0], patch_size[1])
        )
        self.dual_norm_in_patch_embed = dual_norm_in_patch_embed and norm_layer
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()
        self.norm_before = norm_layer(tubelet_size * math.prod(patch_size) * 3) if dual_norm_in_patch_embed and norm_layer else nn.Identity()

    def forward(self, x):
        B, C, T, H, W = x.shape
        x = x.permute(0, 2, 3, 4, 1)
        x = einops.rearrange(x, "b (t1 t2) (ht hp) (wt wp) c -> b (t1 ht wt) (t2 hp wp c)", t2=self.tubelet_size, hp=self.patch_size[0], wp=self.patch_size[1])
        x = self.norm_before(x)
        x = einops.rearrange(x, "b (t1 ht wt) (t2 hp wp c) -> b (t1 t2) (ht hp) (wt wp) c", t1=T // self.tubelet_size, ht=H // self.patch_size[0], t2=self.tubelet_size, hp=self.patch_size[0], wp=self.patch_size[1])
        x = x.permute(0, 4, 1, 2, 3)
        x = self.proj(x)
        x = x.flatten(3).permute(0, 2, 3, 1)
        x = self.norm(x)
        return x


class MLP_Decoder(nn.Module):
    def __init__(self, in_channels=768, out_channels=768,
                 norm_layer=nn.LayerNorm, norm_type='l2'):
        super().__init__()
        self.norm_type = norm_type
        print(f'Normalization Type: {norm_type}')

        self.head = nn.Sequential(
            nn.Linear(in_channels, in_channels),
            nn.GELU(),
            nn.Linear(in_channels, out_channels)
        )
        self.norm = norm_layer(out_channels)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x):
        x = self.norm(self.head(x))

        if self.norm_type == 'l2':
            x = x / x.norm(dim=-1, keepdim=True)
        elif self.norm_type == 'none':
            pass
        else:
            raise NotImplementedError

        return x


class InternVideo2(nn.Module):
    def __init__(
            self,
            in_chans: int = 3,
            patch_size: int = 14,
            img_size: int = 224,
            qkv_bias: bool = False,
            drop_path_rate: float = 0.05,
            embed_dim: int = 384,
            num_heads: int = 6,
            mlp_ratio: float = 4,
            init_values: float = 1e-5,
            qk_normalization: bool = True,
            depth: int = 12,
            use_flash_attn: bool = True,
            use_fused_rmsnorm: bool = True,
            use_fused_mlp: bool = True,
            fused_mlp_heuristic: int = 1,
            attn_pool_num_heads: int = 16,
            clip_embed_dim: int = 768,
            layerscale_no_force_fp32: bool = False,
            num_frames: int = 8,
            tubelet_size: int = 1,
            sep_pos_embed: bool = False,
            use_checkpoint: bool = False,
            checkpoint_num: int = 0,
            cls_token_num: int = 4,
            # for clip
            clip_teacher_embed_dim: int = 3200,
            clip_teacher_final_dim: int = 768,
            clip_norm_type: str = 'l2',
            clip_return_layer: int = 1,
            clip_student_return_interval: int = 1,
            clip_student_return_index: list = None,
            clip_student_decoder: str = 'MLP_Decoder',
            # for diff
            diffloss_d: int = 3,
            diffloss_w: int = 1024,
            num_sampling_steps: str = '1000',
            diffusion_batch_mul: int = 4,
            grad_checkpointing: bool = False,
            use_rope: bool = False,
            # for recovery
            recovery_depth: int = 4,
            recovery_mse: bool = False,
            with_text_init: bool = True,
            diffloss: bool = True,
        ):
        super().__init__()

        self.use_diffloss = diffloss

        assert use_flash_attn == use_fused_rmsnorm == use_fused_mlp, print(
            'use_flash_attn, use_fused_rmsnorm and use_fused_mlp should be consistent')

        self.use_flash_attn = use_flash_attn
        self.embed_dim = embed_dim
        self.cls_token_num = cls_token_num
        self.patch_size = patch_size

        self.clip_norm_type = clip_norm_type
        self.clip_return_index = []
        if clip_student_return_index:
            self.clip_return_index = clip_student_return_index
        else:
            for i in range(clip_return_layer):
                self.clip_return_index.append(depth - int(i * clip_student_return_interval) - 1)
        print(f'CLIP Normalization Type: {clip_norm_type}')
        print(f'CLIP Student Return Index: {self.clip_return_index}')

        if use_fused_rmsnorm:
            norm_layer_for_blocks = partial(DropoutAddRMSNorm, eps=1e-6, prenorm=True)
        else:
            norm_layer_for_blocks = partial(RMSNorm, eps=1e-6)

        self.norm_layer_for_blocks = norm_layer_for_blocks
        self.num_frames = num_frames
        self.patch_embed = PatchEmbed(
            img_size, patch_size, in_chans, embed_dim,
            num_frames=num_frames, tubelet_size=tubelet_size,
            norm_layer=partial(RMSNorm, eps=1e-6), dual_norm_in_patch_embed=True,
        )
        num_patches = self.patch_embed.num_patches
        self.cls_token = nn.Parameter(torch.zeros(1, cls_token_num, embed_dim))

        self.sep_pos_embed = sep_pos_embed
        if sep_pos_embed:
            print("Use seperable position embedding")
            grid_size = self.patch_embed.grid_size
            self.grid_size = grid_size
            self.pos_embed_spatial = nn.Parameter(torch.zeros(1, grid_size[1] * grid_size[2], embed_dim))
            self.pos_embed_temporal = nn.Parameter(torch.zeros(1, grid_size[0], embed_dim))
            self.pos_embed_cls = nn.Parameter(torch.zeros(1, cls_token_num, embed_dim))
            self.clip_pos_embed_spatial = nn.Parameter(torch.zeros(1, grid_size[1] * grid_size[2], embed_dim))
            self.clip_pos_embed_temporal = nn.Parameter(torch.zeros(1, grid_size[0], embed_dim))
            self.clip_pos_embed_cls = nn.Parameter(torch.zeros(1, cls_token_num, embed_dim))
        else:
            print("Use joint position embedding")
            self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + cls_token_num, embed_dim))
            self.clip_pos_embed = nn.Parameter(torch.zeros(1, num_patches + cls_token_num, embed_dim))
            self.diff_pos_embed = nn.Parameter(torch.zeros(1, num_patches + cls_token_num, embed_dim))

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]
        with_cp_list = [False] * depth
        if use_checkpoint:
            for idx in range(depth):
                if idx < checkpoint_num:
                    with_cp_list[idx] = True
        print(f"Droppath rate: {dpr}")
        print(f"Checkpoint list: {with_cp_list}")

        self.blocks = nn.ModuleList([
            Block(
                embed_dim, num_heads, mlp_ratio, qkv_bias=qkv_bias,
                norm_layer=norm_layer_for_blocks,
                drop_path=dpr[i], init_values=init_values, attn_drop=0.,
                use_flash_attn=use_flash_attn, use_fused_mlp=use_fused_mlp,
                fused_mlp_heuristic=fused_mlp_heuristic,
                with_cp=with_cp_list[i],
                use_rope=use_rope,
                qk_normalization=qk_normalization,
                layerscale_no_force_fp32=layerscale_no_force_fp32,
                use_fused_rmsnorm=use_fused_rmsnorm,
                cls_token_num=cls_token_num)
            for i in range(depth)])

        self.clip_projector = AttentionPoolingBlock(
            dim=embed_dim, num_heads=attn_pool_num_heads, qkv_bias=True, qk_scale=None,
            drop=0., attn_drop=0., norm_layer=partial(nn.LayerNorm, eps=1e-5), out_dim=clip_embed_dim)

        # CLIP decoder
        self.clip_decoder = nn.ModuleList([
            eval(clip_student_decoder)(
                in_channels=embed_dim,
                out_channels=clip_teacher_embed_dim,
                norm_layer=partial(nn.LayerNorm, eps=1e-5),
                norm_type=clip_norm_type
            ) for _ in range(clip_return_layer)
        ])

        self.recovery_depth = recovery_depth

        self.to_bert_proj = eval(clip_student_decoder)(
            in_channels=embed_dim,
            out_channels=1024,
            norm_layer=partial(nn.LayerNorm, eps=1e-5),
            norm_type='none'
        )

        self.final_clip_decoder = nn.Identity()
        if clip_teacher_final_dim > 0:
            self.final_clip_decoder = eval(clip_student_decoder)(
                in_channels=clip_embed_dim,
                out_channels=clip_teacher_final_dim,
                norm_layer=partial(nn.LayerNorm, eps=1e-5),
                norm_type=clip_norm_type
            )

        self.bert_mask_token = nn.Parameter(torch.rand((1, 1, embed_dim)), requires_grad=True)

        self.init_pos_embed()

        trunc_normal_(self.cls_token, std=.02)
        trunc_normal_(self.bert_mask_token, std=.02)

        self.apply(self._init_weights)
        self.fix_init_weight()

        if self.use_diffloss:
            self.diffloss = DiffLoss(
                target_channels=in_chans * patch_size * patch_size,
                z_channels=1024,
                width=diffloss_w,
                depth=diffloss_d,
                num_sampling_steps=num_sampling_steps,
                grad_checkpointing=grad_checkpointing
            )
            self.diffusion_batch_mul = diffusion_batch_mul
        else:
            self.mse_projector = eval(clip_student_decoder)(
                in_channels=1024,
                out_channels=in_chans * patch_size * patch_size,
                norm_layer=partial(nn.LayerNorm, eps=1e-5),
                norm_type='none'
            )

        from transformers import ModernBertForMaskedLM, ModernBertConfig

        if not with_text_init:
            self.bert_decoder_config = ModernBertConfig(
                _name_or_path="ModernBERT-large",
                architectures=["ModernBertForMaskedLM"],
                attention_bias=False,
                attention_dropout=0.0,
                bos_token_id=50281,
                classifier_activation="gelu",
                classifier_bias=False,
                classifier_dropout=0.0,
                classifier_pooling="mean",
                cls_token_id=50281,
                decoder_bias=True,
                deterministic_flash_attn=False,
                embedding_dropout=0.0,
                eos_token_id=50282,
                global_attn_every_n_layers=3,
                global_rope_theta=160000.0,
                gradient_checkpointing=False,
                hidden_activation="gelu",
                hidden_size=1024,
                initializer_cutoff_factor=2.0,
                initializer_range=0.02,
                intermediate_size=2624,
                layer_norm_eps=1e-5,
                local_attention=128,
                local_rope_theta=10000.0,
                max_position_embeddings=8192,
                mlp_bias=False,
                mlp_dropout=0.0,
                model_type="modernbert",
                norm_bias=False,
                norm_eps=1e-05,
                num_attention_heads=16,
                num_hidden_layers=28,
                pad_token_id=50283,
                position_embedding_type="absolute",
                sep_token_id=50282,
                tie_word_embeddings=True,
                torch_dtype="float32",
                transformers_version="4.47.0.dev0",
                vocab_size=50368,
            )
            self.bert_decoder = ModernBertForMaskedLM(self.bert_decoder_config).model
        else:
            self.bert_decoder = ModernBertForMaskedLM.from_pretrained(
                'answerdotai/ModernBERT-large',
                dtype=torch.bfloat16
            ).model

        self.bert_decoder.layers = self.bert_decoder.layers[-5:]
        self.bert_decoder.train()

    def init_pos_embed(self):
        print("Init pos_embed from sincos pos_embed")
        if self.sep_pos_embed:
            pos_embed_spatial = get_2d_sincos_pos_embed(
                self.pos_embed_spatial.shape[-1],
                self.patch_embed.grid_size[1],
            )
            self.pos_embed_spatial.data.copy_(torch.from_numpy(pos_embed_spatial).float().unsqueeze(0))
            self.clip_pos_embed_spatial.data.copy_(torch.from_numpy(pos_embed_spatial).float().unsqueeze(0))
            pos_embed_temporal = get_1d_sincos_pos_embed(
                self.pos_embed_spatial.shape[-1],
                self.patch_embed.grid_size[0],
            )
            self.pos_embed_temporal.data.copy_(torch.from_numpy(pos_embed_temporal).float().unsqueeze(0))
            self.clip_pos_embed_temporal.data.copy_(torch.from_numpy(pos_embed_temporal).float().unsqueeze(0))
        else:
            pos_embed = get_3d_sincos_pos_embed(
                self.pos_embed.shape[-1],
                self.patch_embed.grid_size[1],
                self.patch_embed.grid_size[0],
                cls_token=True,
                cls_token_num=self.cls_token_num,
            )
            self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))
            self.clip_pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))
            self.diff_pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
            if m.weight is not None:
                nn.init.constant_(m.weight, 1.0)

    def fix_init_weight(self):
        def rescale(param, layer_id):
            param.div_(math.sqrt(2.0 * layer_id))

        for layer_id, layer in enumerate(self.blocks):
            rescale(layer.attn.proj.weight.data, layer_id + 1)
            rescale(layer.mlp.fc2.weight.data, layer_id + 1)

    @property
    def dtype(self):
        return self.patch_embed.proj.weight.dtype

    def get_num_layers(self):
        return len(self.blocks)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {
            'pos_embed',
            'pos_embed_spatial',
            'pos_embed_temporal',
            'pos_embed_cls',
            'cls_token',
            'clip_pos_embed',
            'clip_pos_embed_spatial',
            'clip_pos_embed_temporal',
            'clip_pos_embed_cls',
        }

    def forward_diff_loss(self, z, target):

        with torch.cuda.amp.autocast(dtype=torch.float32):
            _, _, C_target = target.shape
            _, _, C_z = z.shape

            target = target.reshape(-1, C_target).repeat(self.diffusion_batch_mul, 1)
            z = z.reshape(-1, C_z).repeat(self.diffusion_batch_mul, 1)

            loss = self.diffloss(z=z, target=target)
        return loss

    def transform_tensor(self, x, patch_size):
        B, C, T, H, W = x.shape
        x = x.permute(0, 2, 1, 3, 4)
        x = x.reshape(B * T, C, H, W)
        BT, C, H, W = x.shape
        x = F.pixel_unshuffle(x, patch_size).permute(0, 2, 3, 1)
        C = x.shape[-1]
        x = x.reshape(B, T, -1, C).reshape(B, -1, C)
        return x

    def forward_reconstruction_loss(self, x, to_reconstruct, diff_pos_embed, cls_token_num, mask, mask_target, B, T, L, C):
        x_input_diff = x[:, cls_token_num:, :]
        bert_mask_token = self.bert_mask_token.repeat(B, T * L, 1)
        bert_mask_token[~mask] = bert_mask_token[~mask] * 0. + x_input_diff.flatten(0, 1)
        bert_mask_token = bert_mask_token + diff_pos_embed[:, cls_token_num:, :].repeat(B, 1, 1)

        C_bert = bert_mask_token.shape[-1]
        mask_target_loss_compute = mask[~mask_target].reshape(B, -1)
        forward_bert_mask_token = bert_mask_token[~mask_target].contiguous().reshape(B, -1, C_bert)

        forward_bert_mask_token = torch.cat(
            (x[:, :cls_token_num, :] + self.diff_pos_embed[:, :cls_token_num, :].repeat(B, 1, 1), forward_bert_mask_token),
            dim=1)

        B_bert, N_bert, C_bert = forward_bert_mask_token.shape
        forward_bert_mask_token = self.to_bert_proj(forward_bert_mask_token.flatten(0, 1))

        attn_mask = torch.ones_like(forward_bert_mask_token, dtype=torch.bool)
        max_seqlen = N_bert
        cu_seqlens = torch.arange(B_bert + 1, dtype=torch.int32, device=forward_bert_mask_token.device) * N_bert

        for layer in self.bert_decoder.layers[-5:]:
            forward_bert_mask_token = layer(
                hidden_states=forward_bert_mask_token,
                attention_mask=attn_mask,
                cu_seqlens=cu_seqlens,
                max_seqlen=max_seqlen
            )[0]

        forward_bert_mask_token = forward_bert_mask_token.reshape(B_bert, -1, forward_bert_mask_token.shape[-1])
        forward_bert_mask_token = forward_bert_mask_token[:, cls_token_num:, :]

        to_reconstruct = mask_with_cls(mask_target, self.transform_tensor(to_reconstruct, self.patch_size), with_cls=False)

        loss_diff = self.forward_diff_loss(
            forward_bert_mask_token[mask_target_loss_compute].reshape(B, -1, forward_bert_mask_token.shape[-1]),
            to_reconstruct[mask_target_loss_compute].reshape(B, -1, to_reconstruct.shape[-1])
        )

        return loss_diff

    def expand_pos_embed(self, pos_embed, new_t_size, L, num_extra_tokens=-1):
        if num_extra_tokens == -1:
            num_extra_tokens = self.cls_token_num

        pos_embed_checkpoint = pos_embed
        embedding_size = pos_embed_checkpoint.shape[-1]
        orig_size = int(((pos_embed_checkpoint.shape[-2] - num_extra_tokens) // (self.num_frames / self.patch_embed.tubelet_size)) ** 0.5)
        new_size = int(L ** 0.5)

        if self.num_frames != new_t_size:
            extra_tokens = pos_embed_checkpoint[:, :num_extra_tokens]
            pos_tokens = pos_embed_checkpoint[:, num_extra_tokens:]
            pos_tokens = pos_tokens.reshape(1, self.num_frames, -1, embedding_size)
            pos_tokens = pos_tokens.permute(0, 2, 3, 1).reshape(-1, embedding_size, self.num_frames)
            pos_tokens = torch.nn.functional.interpolate(pos_tokens.cpu(), size=new_t_size, mode='linear').cuda()
            pos_tokens = pos_tokens.reshape(1, -1, embedding_size, new_t_size)
            pos_tokens = pos_tokens.permute(0, 3, 1, 2).reshape(1, -1, embedding_size)
            new_pos_embed = torch.cat((extra_tokens, pos_tokens), dim=1)
            pos_embed_checkpoint = new_pos_embed

        if orig_size != new_size:
            extra_tokens = pos_embed_checkpoint[:, :num_extra_tokens]
            pos_tokens = pos_embed_checkpoint[:, num_extra_tokens:]
            pos_tokens = pos_tokens.reshape(-1, new_t_size, orig_size, orig_size, embedding_size)
            pos_tokens = pos_tokens.reshape(-1, orig_size, orig_size, embedding_size).permute(0, 3, 1, 2)
            pos_tokens = torch.nn.functional.interpolate(
                pos_tokens.cpu(), size=(new_size, new_size), mode='bicubic', align_corners=False).cuda()
            pos_tokens = pos_tokens.permute(0, 2, 3, 1).reshape(-1, new_t_size, new_size, new_size, embedding_size)
            pos_tokens = pos_tokens.flatten(1, 3)
            new_pos_embed = torch.cat((extra_tokens, pos_tokens), dim=1)

        return new_pos_embed

    def dynamic_masking(self, x, B, T, L, C, mask_ratio):
        patch_embed_vectors = x.detach().clone()

        if T == 1:
            k = int(T * L * mask_ratio)
            rand_indices = torch.rand(B, T * L, device=x.device).argsort(dim=1)
            mask = torch.zeros(B, T * L, device=x.device)
            mask.scatter_(1, rand_indices[:, :k], 1)
            mask = mask.bool()
            return mask

        use_sparse_block = False
        orig_T = T
        if orig_T > 4 and orig_T % 2 == 0:
            use_sparse_block = True
            mod4 = False
            if T % 4 == 0:
                mod4 = True
                patch_embed_vectors = patch_embed_vectors.reshape(B, 4, T // 4, L, C).reshape(B * 4, T // 4, L, C)
                T, B = T // 4, B * 4
            else:
                patch_embed_vectors = patch_embed_vectors.reshape(B, 2, T // 2, L, C).reshape(B * 2, T // 2, L, C)
                T, B = T // 2, B * 2

        distance = torch.norm(patch_embed_vectors[:, :T - 1, :, :] - patch_embed_vectors[:, 1:, :, :], p=2, dim=3)
        importance = torch.cat((distance[:, 0, :], distance.flatten(1)), dim=1)
        ids_sorted = torch.argsort(importance, dim=1, descending=True)
        num_input_tokens = int((1 - mask_ratio) * (T * L))

        ids_restore = torch.argsort(ids_sorted, dim=1)
        input_mask = torch.ones([B, T * L], device=x.device)
        input_mask[:, :num_input_tokens] = 0

        input_mask = torch.gather(input_mask, dim=1, index=ids_restore)

        if use_sparse_block:
            if mod4:
                T, B = T * 4, B // 4
            else:
                T, B = T * 2, B // 2
            input_mask = input_mask.reshape(B, T * L)

        return input_mask.to(torch.bool)

    def forward(self, x, mask=None, mask_target=None, mask_ratio=0., embed_only=False):
        cls_token_num = self.cls_token_num
        to_reconstruct = x.clone().detach()
        x = self.patch_embed(x.type(self.dtype))
        B, T, L, C = x.shape

        if mask is None and mask_ratio != 0.:
            mask = self.dynamic_masking(x, B, T, L, C, mask_ratio=mask_ratio)

        if mask_target is None and mask is not None:
            mask_target = torch.zeros_like(mask).bool()

        x = x.view([B, T * L, C])

        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)

        target_shape = x[0].shape
        pos_embed = self.pos_embed
        clip_pos_embed = self.clip_pos_embed
        diff_pos_embed = self.diff_pos_embed

        if self.pos_embed[0].shape != target_shape:
            pos_embed = self.expand_pos_embed(self.pos_embed, T, L)
            clip_pos_embed = self.expand_pos_embed(self.clip_pos_embed, T, L)
            diff_pos_embed = self.expand_pos_embed(self.diff_pos_embed, T, L)

        x = x + pos_embed

        if mask is not None:
            cls_tokens, x = x[:, :cls_token_num, :].reshape(B, -1, C), x[:, cls_token_num:, :].reshape(B, -1, C)
            x = x[~mask].reshape(B, -1, C)
            x = torch.cat((cls_tokens, x), dim=1)

        residual = None
        x_clip = []
        for idx, blk in enumerate(self.blocks):
            if isinstance(x, tuple) and len(x) == 2:
                x, residual = x
            x = blk(x, residual=residual, mask=mask)

            if idx == self.recovery_depth:
                recovery_x = x
                if isinstance(x, tuple) and len(x) == 2:
                    recovery_x, _ = recovery_x

            if idx in self.clip_return_index:
                if isinstance(x, tuple) and len(x) == 2:
                    tmp_x, tmp_residual = x
                    if tmp_residual is not None:
                        x_clip.append(tmp_x + tmp_residual)
                else:
                    x_clip.append(x)

        if isinstance(x, tuple) and len(x) == 2:
            x, residual = x
            if residual is not None:
                x = x + residual

        if embed_only:
            return x, self.clip_projector(x)

        reconstruction_loss = self.forward_reconstruction_loss(x, to_reconstruct, diff_pos_embed, cls_token_num, mask, mask_target, B, T, L, C)

        # align CLIP
        x_clip = torch.stack(x_clip)
        K, B_clip, _, C_CLIP = x_clip.shape

        clip_pos_embed = clip_pos_embed.repeat(B_clip, 1, 1)

        cls_tokens_clip, clip_pos_embed = clip_pos_embed[:, :cls_token_num, :].reshape(B_clip, -1, C_CLIP), clip_pos_embed[:, cls_token_num:, :].reshape(B_clip, -1, C_CLIP)
        clip_pos_embed = clip_pos_embed[~mask].reshape(B_clip, -1, C_CLIP)
        clip_pos_embed = torch.cat((cls_tokens_clip, clip_pos_embed), dim=1)

        x_clip = x_clip + clip_pos_embed.unsqueeze(0).repeat(K, 1, 1, 1)

        x_clip_align = []
        for clip_decoder in self.clip_decoder:
            x_clip_align.append(clip_decoder(x_clip[idx][:, cls_token_num:, :].view(B_clip, -1, C_CLIP)))
        x_clip_align = torch.stack(x_clip_align)

        x_align = self.final_clip_decoder(self.clip_projector(x))

        if mask_ratio != 0.:
            return x_clip_align, x_align, reconstruction_loss, mask

        return x_clip_align, x_align, reconstruction_loss


@register_model
def internvideo_next_stage1_base(pretrained=False, **kwargs):
    model = InternVideo2(
        img_size=224, patch_size=14, embed_dim=768,
        depth=12, num_heads=12, mlp_ratio=4,
        attn_pool_num_heads=16, clip_embed_dim=768, recovery_mse=True,
        diffloss_d=6, diffloss_w=1536, diffloss=True, diffusion_batch_mul=1,
        **kwargs
    )
    return model


@register_model
def internvideo_next_stage1_large(pretrained=False, **kwargs):
    model = InternVideo2(
        img_size=224, patch_size=14, embed_dim=1024,
        depth=24, num_heads=16, mlp_ratio=4,
        attn_pool_num_heads=16, clip_embed_dim=768, recovery_mse=True,
        diffloss_d=6, diffloss_w=1536, diffloss=True, diffusion_batch_mul=1,
        **kwargs
    )
    return model


@register_model
def internvideo_next_stage1_1b(pretrained=False, **kwargs):
    model = InternVideo2(
        img_size=224, patch_size=14, embed_dim=1408,
        depth=40, num_heads=16, mlp_ratio=48 / 11,
        attn_pool_num_heads=16, clip_embed_dim=768, recovery_mse=True,
        diffloss_d=6, diffloss_w=1536, diffloss=True, diffusion_batch_mul=1,
        **kwargs
    )
    return model
