#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2024 Apple Inc. All Rights Reserved.
#
import math
from typing import Optional, Sequence

import torch
from torch import Tensor, nn

from .transformer import (
    PositionalEmbedding,
    TransformerEncoder,
    get_normalization_layer,
)


class TextTransformer(nn.Module):
    def __init__(self, cfg: dict, projection_dim: int, *args, **kwargs) -> None:
        super().__init__()

        model_dim = cfg["dim"]
        no_scale_embedding = cfg.get("no_scale_embedding", False)
        no_pos_embedding = cfg.get("no_pos_embedding", False)
        embed_dropout = cfg.get("embed_dropout", 0.0)
        norm_layer = cfg["norm_layer"]
        variant = cfg["model_name"]
        self.vocab_size = cfg["vocab_size"]
        self.projection_dim = projection_dim

        # Token embedding layer
        self.embedding_layer = nn.Embedding(
            embedding_dim=model_dim, num_embeddings=self.vocab_size
        )
        self.embed_scale = 1.0 if no_scale_embedding else model_dim**-0.5

        # Context length
        context_length = cfg["context_length"]
        assert (
            context_length is not None
        ), "Context length can't be None. Please set value accordingly."

        self.positional_embedding = (
            None
            if no_pos_embedding
            else PositionalEmbedding(
                num_embeddings=context_length, embedding_dim=model_dim
            )
        )

        self.embedding_dropout = nn.Dropout(p=embed_dropout)

        # Transformer layer
        n_transformer_layers = cfg["n_transformer_layers"]

        # FFN multipliers for transformer layer
        ffn_multipliers = cfg["ffn_multiplier_per_layer"]
        if isinstance(ffn_multipliers, (float, int)):
            ffn_multipliers = [ffn_multipliers] * n_transformer_layers

        if not isinstance(ffn_multipliers, Sequence):
            Warning(
                "{} expects FFN multipliers as a list, whose length is the same as"
                " number of transformer layers. Got: {}".format(
                    self.__class__.__name__, type(ffn_multipliers)
                )
            )
        elif (
            isinstance(ffn_multipliers, Sequence)
            and len(ffn_multipliers) != n_transformer_layers
        ):
            Warning(
                "We need FFN multiplier for each transformer layer. Got {} ffn"
                " multipliers while number of transformer layers = {}".format(
                    len(ffn_multipliers), n_transformer_layers
                )
            )
        ffn_dims = [
            int(math.ceil(model_dim * ffn_mult / 16.0) * 16.0)
            for ffn_mult in ffn_multipliers
        ]

        # Heads for transformer layers
        mha_heads = cfg["n_heads_per_layer"]
        if isinstance(mha_heads, int):
            mha_heads = [mha_heads] * n_transformer_layers

        if not isinstance(mha_heads, Sequence):
            Warning(
                "{} expects MHA heads as a list, whose length is the same as number of "
                "transformer layers. Got: {}".format(
                    self.__class__.__name__, type(mha_heads)
                )
            )
        elif isinstance(mha_heads, Sequence) and len(mha_heads) != n_transformer_layers:
            Warning(
                "{} needs MHA heads for each transformer layer. Got {} mha heads while"
                " number of transformer layers = {}".format(
                    self.__class__.__name__, len(mha_heads), n_transformer_layers
                )
            )

        if variant == "base":
            self.transformer = nn.ModuleList(
                [
                    TransformerEncoder(
                        embed_dim=model_dim,
                        num_heads=mha_heads[layer_idx],
                        ffn_latent_dim=ffn_dims[layer_idx],
                        transformer_norm_layer=norm_layer,
                    )
                    for layer_idx in range(n_transformer_layers)
                ]
            )
        elif variant == "mct":
            raise NotImplementedError
        else:
            raise ValueError("Unrecognized text encoder variant {}".format(variant))

        self.final_layer_norm = get_normalization_layer(
            num_features=model_dim, norm_type=norm_layer
        )

        self.projection_layer = nn.Parameter(
            torch.empty(model_dim, self.projection_dim)
        )
        self.model_dim = model_dim
        self.causal_masking = cfg["causal_masking"]

    def forward_embedding(self, text_tokens: Tensor) -> Tensor:
        """Return text embedding for all tokens.

        Args:
            text_tokens: a tensor of token indices. Shape: [batch_size, context_length]

        Returns:
            A tensor of [batch_size, context_length, hidden_dim].
        """
        # [batch_size, context_length] --> [batch_size, context_length, hidden_dim]
        token_emb = self.embedding_layer(text_tokens)
        seq_len = token_emb.shape[1]
        if self.positional_embedding is not None:
            token_emb = token_emb + self.positional_embedding(seq_len).to(
                token_emb.dtype
            )
        token_emb = self.embedding_dropout(token_emb)
        return token_emb

    def build_attention_mask(self, context_length: int, batch_size: int) -> Tensor:
        """Build causal attention mask [batch_size, context_length, context_length]."""
        # Build mask with full attention between the tokens
        # pytorch uses additive attention mask; fill with -inf
        mask = torch.empty(context_length, context_length)
        mask.fill_(float("-inf"))
        mask.triu_(1)  # zero out the lower diagonal
        mask = mask.unsqueeze(0)  # add dummy batch dimension
        mask = mask.expand(batch_size, -1, -1)
        return mask

    def encode_text(
        self,
        text_tokens: Tensor,
        key_padding_mask: Optional[Tensor] = None,
        return_all_tokens: bool = False,
        *args,
        **kwargs
    ) -> Tensor:
        """Return text token embeddings.

        Args:
            text_tokens: a tensor of token indices. Shape: [batch_size, context_length]
            key_padding_mask: a tensor of boolean values as the padding mask.
                Shape: [batch_size, context_length]
            return_all_tokens: a boolean flag to return all tokens, defaults to False
                to return only EOT token embedding.
        Returns:
            A tensor of [batch_size, context_length, hidden_dim] if return_all_tokens is
            True, otherwise a tensor of [batch_size, hidden_dim].
        """
        # Discrete tokens to continuous embeddings
        # [batch_size, context_length] --> [batch_size, context_length, hidden_dim]
        token_emb = self.forward_embedding(text_tokens)

        # [1, context_length, context_length]
        attn_mask = None
        if self.causal_masking:
            attn_mask = self.build_attention_mask(
                context_length=text_tokens.shape[1], batch_size=text_tokens.shape[0]
            )
            attn_mask = attn_mask.to(device=token_emb.device, dtype=token_emb.dtype)
            key_padding_mask = None

        for layer in self.transformer:
            token_emb = layer(
                token_emb,
                key_padding_mask=key_padding_mask,
                attn_mask=attn_mask,
            )

        # Apply layer norm
        token_emb = self.final_layer_norm(token_emb)

        if return_all_tokens:
            return token_emb

        # Take features from the eot embedding (eot_token is the highest number in each sequence)
        token_emb = token_emb[
            torch.arange(text_tokens.shape[0]), text_tokens.argmax(dim=-1)
        ]

        token_emb = token_emb @ self.projection_layer
        return token_emb

    def forward(
        self,
        text_tokens: Tensor,
        key_padding_mask: Optional[Tensor] = None,
        return_all_tokens: bool = False,
        *args,
        **kwargs
    ) -> Tensor:
        # Image-text pair data with single caption
        # [B, CL] --> [B, d]
        text_tokens = self.encode_text(
            text_tokens=text_tokens,
            key_padding_mask=key_padding_mask,
            return_all_tokens=return_all_tokens,
            *args,
            **kwargs
        )
        return text_tokens
