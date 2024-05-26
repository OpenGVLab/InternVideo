import torch
import torch.nn as nn
import torch.nn.functional as F
from easydict import EasyDict as edict
from xml.model_components import BertAttention, TrainablePositionalEncoding


class TextEncoder(nn.Module):
    def __init__(self, hidden_size, drop, input_drop, nheads, max_position_embeddings):
        super().__init__()
        self.transformer_encoder = BertAttention(edict(
            hidden_size=hidden_size,
            intermediate_size=hidden_size,
            hidden_dropout_prob=drop,
            attention_probs_dropout_prob=drop,
            num_attention_heads=nheads,
        ))
        self.pos_embed = TrainablePositionalEncoding(
            max_position_embeddings=max_position_embeddings,
            hidden_size=hidden_size,
            dropout=input_drop,
        )
        self.modular_vector_mapping = nn.Linear(hidden_size, 1, bias=False)

    def forward(self, feat, mask):
        """
        Args:
            feat: (N, L, D=hidden_size)
            mask: (N, L) with 1 indicates valid

        Returns:
            (N, D)
        """
        feat = self.pos_embed(feat)  # (N, L, D)
        feat = self.transformer_encoder(feat, mask.unsqueeze(1))
        att_scores = self.modular_vector_mapping(feat)  # (N, L, 1)
        att_scores = F.softmax(mask_logits(att_scores, mask.unsqueeze(2)), dim=1)
        pooled_feat = torch.einsum("blm,bld->bmd", att_scores, feat)  # (N, 2 or 1, D)
        return pooled_feat.squeeze(1)


def mask_logits(target, mask):
    return target * mask + (1 - mask) * (-1e10)


def build_text_encoder(args):
    return TextEncoder(
        hidden_size=args.hidden_dim,
        drop=args.dropout,
        input_drop=args.input_dropout,
        nheads=args.nheads,
        max_position_embeddings=args.max_q_l
    )
