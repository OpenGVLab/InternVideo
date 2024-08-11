#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2024 Apple Inc. All Rights Reserved.
#
"""
Implementation of the following modules is borrowed from ml-cvnets repo:
https://github.com/apple/ml-cvnets/blob/main/cvnets/layers/multi_head_attention.py
https://github.com/apple/ml-cvnets/blob/main/cvnets/text_encoders/transformer.py

Please see ACKNOWLEDGEMENTS for license details.
"""

from typing import List, Optional, Union

import torch
from torch import Size, Tensor, nn
from torch.nn import functional as F
from torchvision.ops import StochasticDepth


class LayerNormFP32(nn.LayerNorm):
    """
    Applies `Layer Normalization <https://arxiv.org/abs/1607.06450>`_ over a input tensor with FP32 precision
    """

    def __init__(
        self,
        normalized_shape: Union[int, List[int], Size],
        eps: Optional[float] = 1e-5,
        elementwise_affine: Optional[bool] = True,
        *args,
        **kwargs,
    ):
        super().__init__(
            normalized_shape=normalized_shape,
            eps=eps,
            elementwise_affine=elementwise_affine,
            *args,
            **kwargs,
        )

    def forward(self, x: Tensor) -> Tensor:
        # Convert input from dtype X to FP32 and perform normalization operation.
        # This may help with underflow/overflow issues that we typically see with normalization layers
        inp_dtype = x.dtype
        return super().forward(x.to(torch.float32)).to(inp_dtype)


def get_normalization_layer(norm_type, num_features):
    if norm_type == "layer_norm":
        return nn.LayerNorm(num_features)
    elif norm_type == "layer_norm_fp32":
        return LayerNormFP32(num_features)
    else:
        raise NotImplementedError(f"Option: {norm_type} not supported.")


class PositionalEmbedding(nn.Module):
    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        padding_idx: Optional[int] = None,
        is_learnable: Optional[bool] = False,
        interpolation_mode: Optional[str] = "bilinear",
        *args,
        **kwargs,
    ):
        super().__init__()
        # Add other pos embedding here and logic to choose between them
        module = LearnablePositionalEmbedding

        self.pos_embed = module(
            num_embeddings=num_embeddings,
            embedding_dim=embedding_dim,
            padding_idx=padding_idx,
            interpolation_mode=interpolation_mode,
            *args,
            **kwargs,
        )

    def forward(self, seq_len: int, *args, **kwargs) -> Tensor:
        return self.pos_embed(seq_len, *args, **kwargs)

    def __repr__(self):
        return self.pos_embed.__repr__()


class LearnablePositionalEmbedding(nn.Module):
    """Learnable Positional embedding"""

    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        padding_idx: Optional[int] = None,
        interpolation_mode: Optional[str] = "bilinear",
        *args,
        **kwargs,
    ):
        super().__init__()
        self.pos_embed = nn.Parameter(torch.empty(1, 1, num_embeddings, embedding_dim))
        self.embedding_dim = embedding_dim
        self.num_embeddings = num_embeddings
        self.padding_idx = padding_idx
        self.interpolation_mode = interpolation_mode

        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.trunc_normal_(self.pos_embed, mean=0, std=self.embedding_dim**-0.5)
        if self.padding_idx is not None:
            with torch.no_grad():
                self.pos_embed[:, :, self.padding_idx, ...] = 0.0

    def forward(self, seq_len: int, *args, **kwargs) -> Tensor:
        # scale pos embedding
        pos_embed = self.pos_embed
        if self.padding_idx is not None:
            with torch.no_grad():
                pos_embed[:, :, self.padding_idx, ...] = 0.0

        if seq_len != self.num_embeddings:
            pos_embed = F.interpolate(
                pos_embed,
                size=(seq_len, self.embedding_dim),
                mode=self.interpolation_mode,
            )

        # Input is of the form [Batch, Seq_len, Embedding_dim]
        return pos_embed.reshape(1, seq_len, self.embedding_dim)

    def __repr__(self):
        return "{}(num_embeddings={}, embedding_dim={}, padding_idx={})".format(
            self.__class__.__name__,
            self.num_embeddings,
            self.embedding_dim,
            self.padding_idx,
        )


class MultiHeadAttention(nn.Module):
    """
    This layer applies a multi-head self- or cross-attention as described in
    `Attention is all you need <https://arxiv.org/abs/1706.03762>`_ paper

    Args:
        embed_dim (int): :math:`C_{in}` from an expected input of size :math:`(N, S, C_{in})`
        num_heads (int): Number of heads in multi-head attention
        attn_dropout (Optional[float]): Attention dropout. Default: 0.0
        bias (Optional[bool]): Use bias or not. Default: ``True``

    Shape:
        - Input:
           - Query tensor (x_q) :math:`(N, S, C_{in})` where :math:`N` is batch size, :math:`S` is number of source tokens,
        and :math:`C_{in}` is input embedding dim
           - Optional Key-Value tensor (x_kv) :math:`(N, T, C_{in})` where :math:`T` is number of target tokens
        - Output: same shape as the input

    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        attn_dropout: Optional[float] = 0.0,
        bias: Optional[bool] = True,
        output_dim: Optional[int] = None,
        *args,
        **kwargs,
    ) -> None:
        if output_dim is None:
            output_dim = embed_dim
        super().__init__()
        if embed_dim % num_heads != 0:
            Warning(
                "Embedding dim must be divisible by number of heads in {}. Got: embed_dim={} and num_heads={}".format(
                    self.__class__.__name__, embed_dim, num_heads
                )
            )

        self.qkv_proj = nn.Linear(
            in_features=embed_dim, out_features=3 * embed_dim, bias=bias
        )

        self.attn_dropout = nn.Dropout(p=attn_dropout)
        self.out_proj = nn.Linear(
            in_features=embed_dim, out_features=output_dim, bias=bias
        )

        self.head_dim = embed_dim // num_heads
        self.scaling = self.head_dim**-0.5
        self.softmax = nn.Softmax(dim=-1)
        self.num_heads = num_heads
        self.embed_dim = embed_dim
        self.use_separate_proj_weight = embed_dim != output_dim

    def __repr__(self):
        return "{}(head_dim={}, num_heads={}, attn_dropout={})".format(
            self.__class__.__name__, self.head_dim, self.num_heads, self.attn_dropout.p
        )

    def _forward_impl(
        self,
        x_q: Tensor,
        x_kv: Optional[Tensor] = None,
        key_padding_mask: Optional[Tensor] = None,
        attn_mask: Optional[Tensor] = None,
    ) -> Tensor:
        # [N, S, C]
        b_sz, S_len, in_channels = x_q.shape

        if x_kv is None:
            # self-attention
            # [N, S, C] --> [N, S, 3C] --> [N, S, 3, h, c] where C = hc
            qkv = self.qkv_proj(x_q).reshape(b_sz, S_len, 3, self.num_heads, -1)
            # [N, S, 3, h, c] --> [N, h, 3, S, C]
            qkv = qkv.transpose(1, 3).contiguous()

            # [N, h, 3, S, C] --> [N, h, S, C] x 3
            query, key, value = qkv[:, :, 0], qkv[:, :, 1], qkv[:, :, 2]
        else:
            T_len = x_kv.shape[1]

            # cross-attention
            # [N, S, C]
            query = F.linear(
                x_q,
                weight=self.qkv_proj.weight[: self.embed_dim, ...],
                bias=self.qkv_proj.bias[: self.embed_dim]
                if self.qkv_proj.bias is not None
                else None,
            )
            # [N, S, C] --> [N, S, h, c] --> [N, h, S, c]
            query = (
                query.reshape(b_sz, S_len, self.num_heads, self.head_dim)
                .transpose(1, 2)
                .contiguous()
            )

            # [N, T, C] --> [N, T, 2C]
            kv = F.linear(
                x_kv,
                weight=self.qkv_proj.weight[self.embed_dim :, ...],
                bias=self.qkv_proj.bias[self.embed_dim :]
                if self.qkv_proj.bias is not None
                else None,
            )
            # [N, T, 2C] --> [N, T, 2, h, c]
            kv = kv.reshape(b_sz, T_len, 2, self.num_heads, self.head_dim)
            # [N, T, 2, h, c] --> [N, h, 2, T, c]
            kv = kv.transpose(1, 3).contiguous()
            key, value = kv[:, :, 0], kv[:, :, 1]

        query = query * self.scaling

        # [N h, T, c] --> [N, h, c, T]
        key = key.transpose(-1, -2)

        # QK^T
        # [N, h, S, c] x [N, h, c, T] --> [N, h, S, T]
        attn = torch.matmul(query, key)

        batch_size, num_heads, num_src_tokens, num_tgt_tokens = attn.shape
        if attn_mask is not None:
            # attn_mask shape should be the same as attn
            assert list(attn_mask.shape) == [
                batch_size,
                num_src_tokens,
                num_tgt_tokens,
            ], "Shape of attention mask should be [{}, {}, {}]. Got: {}".format(
                batch_size, num_src_tokens, num_tgt_tokens, attn_mask.shape
            )
            # [N, S, T] --> [N, 1, S, T]
            attn_mask = attn_mask.unsqueeze(1)
            attn = attn + attn_mask

        if key_padding_mask is not None:
            # Do not attend to padding positions
            # key padding mask size is [N, T]
            assert key_padding_mask.dim() == 2 and list(key_padding_mask.shape) == [
                batch_size,
                num_tgt_tokens,
            ], "Key_padding_mask should be 2-dimension with shape [{}, {}]. Got: {}".format(
                batch_size, num_tgt_tokens, key_padding_mask.shape
            )
            attn = attn.masked_fill(
                key_padding_mask.unsqueeze(1)
                .unsqueeze(2)
                .to(torch.bool),  # [N, T] --> [N, 1, 1, T]
                float("-inf"),
            )

        attn_dtype = attn.dtype
        attn_as_float = self.softmax(attn.float())
        attn = attn_as_float.to(attn_dtype)
        attn = self.attn_dropout(attn)

        # weighted sum
        # [N, h, S, T] x [N, h, T, c] --> [N, h, S, c]
        out = torch.matmul(attn, value)

        # [N, h, S, c] --> [N, S, h, c] --> [N, S, C]
        out = out.transpose(1, 2).reshape(b_sz, S_len, -1)
        out = self.out_proj(out)

        return out

    def forward(
        self,
        x_q: Tensor,
        x_kv: Optional[Tensor] = None,
        key_padding_mask: Optional[Tensor] = None,
        attn_mask: Optional[Tensor] = None,
        *args,
        **kwargs,
    ) -> Tensor:
        # [Batch , Sequence, Hidden_dim]
        return self._forward_impl(
            x_q=x_q,
            x_kv=x_kv,
            key_padding_mask=key_padding_mask,
            attn_mask=attn_mask,
        )


class TransformerEncoder(nn.Module):
    """
    This class defines the pre-norm `Transformer encoder <https://arxiv.org/abs/1706.03762>`_
    Args:
        embed_dim: :math:`C_{in}` from an expected input of size :math:`(N, P, C_{in})`.
        ffn_latent_dim: Inner dimension of the FFN.
        num_heads: Number of heads in multi-head attention. Default: 8.
        attn_dropout: Dropout rate for attention in multi-head attention. Default: 0.0
        dropout: Dropout rate. Default: 0.0.
        ffn_dropout: Dropout between FFN layers. Default: 0.0.
        transformer_norm_layer: Normalization layer. Default: layer_norm.
        stochastic_dropout: Stochastic dropout setting. Default: 0.0.

    Shape:
        - Input: :math:`(N, P, C_{in})` where :math:`N` is batch size, :math:`P` is number of patches,
        and :math:`C_{in}` is input embedding dim
        - Output: same shape as the input
    """

    def __init__(
        self,
        embed_dim: int,
        ffn_latent_dim: int,
        num_heads: Optional[int] = 8,
        attn_dropout: Optional[float] = 0.0,
        dropout: Optional[float] = 0.0,
        ffn_dropout: Optional[float] = 0.0,
        transformer_norm_layer: Optional[str] = "layer_norm",
        stochastic_dropout: Optional[float] = 0.0,
        *args,
        **kwargs,
    ) -> None:

        super().__init__()

        # Build attention layer
        attn_unit = MultiHeadAttention(
            embed_dim,
            num_heads,
            attn_dropout=attn_dropout,
            bias=True,
        )

        self.pre_norm_mha = nn.Sequential(
            get_normalization_layer(
                norm_type=transformer_norm_layer, num_features=embed_dim
            ),
            attn_unit,
            nn.Dropout(p=dropout),
        )

        act_name = nn.GELU()
        self.pre_norm_ffn = nn.Sequential(
            get_normalization_layer(
                norm_type=transformer_norm_layer, num_features=embed_dim
            ),
            nn.Linear(in_features=embed_dim, out_features=ffn_latent_dim, bias=True),
            act_name,
            nn.Dropout(p=ffn_dropout),
            nn.Linear(in_features=ffn_latent_dim, out_features=embed_dim, bias=True),
            nn.Dropout(p=dropout),
        )

        self.drop_path = nn.Identity()
        if stochastic_dropout > 0.0:
            if dropout > 0.0:
                Warning(
                    "Stochastic dropout and dropout are mutually exclusive. "
                    "Use either of them, but not both."
                    "Got: {} and {}".format(stochastic_dropout, dropout)
                )
            self.drop_path = StochasticDepth(p=stochastic_dropout, mode="row")

        self.embed_dim = embed_dim
        self.ffn_dim = ffn_latent_dim
        self.ffn_dropout = ffn_dropout
        self.stochastic_dropout = stochastic_dropout
        self.std_dropout = dropout
        self.attn_fn_name = attn_unit.__class__.__name__
        self.act_fn_name = act_name.__class__.__name__
        self.norm_type = transformer_norm_layer

    def __repr__(self) -> str:
        return "{}(embed_dim={}, ffn_dim={}, dropout={}, ffn_dropout={}, stochastic_dropout={}, attn_fn={}, act_fn={}, norm_fn={})".format(
            self.__class__.__name__,
            self.embed_dim,
            self.ffn_dim,
            self.std_dropout,
            self.ffn_dropout,
            self.stochastic_dropout,
            self.attn_fn_name,
            self.act_fn_name,
            self.norm_type,
        )

    def forward(
        self,
        x: Tensor,
        x_prev: Optional[Tensor] = None,
        key_padding_mask: Optional[Tensor] = None,
        attn_mask: Optional[Tensor] = None,
        *args,
        **kwargs,
    ) -> Tensor:

        # Multi-head attention
        res = x
        x = self.pre_norm_mha[0](x)  # norm
        x = self.pre_norm_mha[1](
            x_q=x,
            x_kv=x_prev,
            key_padding_mask=key_padding_mask,
            attn_mask=attn_mask,
            *args,
            **kwargs,
        )  # mha

        x = self.drop_path(self.pre_norm_mha[2](x))  # applying stochastic depth
        x = x + res

        # Feed forward network
        x = x + self.drop_path(self.pre_norm_ffn(x))
        return x
