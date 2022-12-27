import torch
import torch.nn as nn

from .evl_utils.evl_module import ResidualDecoderBlock
from .coca import Residual, ParallelTransformerBlock, CrossAttention
from einops import repeat


class CaptionDecoder(nn.Module):
    def __init__(
        self,
        n_layers,
        transformer_width,
        vision_width,
        transformer_heads,
        vocab_size,
        num_visual_queries=256,
    ):
        super().__init__()
        scale = transformer_width ** -0.5
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
        for attn_ff, cross_attn in self.multimodal_layers:
            text_embeds = attn_ff(text_embeds)
            text_embeds = cross_attn(text_embeds, image_feats)

        logits = self.predictor(text_embeds)

        return logits


class MaskedTextDecoder(nn.Module):
    def __init__(
        self,
        n_layers,
        transformer_width,
        vision_width,
        transformer_heads,
        vocab_size,
        drop_rate,
        drop_path_rate,
    ):
        super().__init__()
        self.visual_encoder_to_decoder = nn.Sequential(
            nn.LayerNorm(vision_width), nn.Linear(vision_width, transformer_width)
        )
        self.text_encoder_to_decoder = nn.Sequential(
            nn.LayerNorm(transformer_width),
            nn.Linear(transformer_width, transformer_width),
        )
        dpr = [
            x.item() for x in torch.linspace(0, drop_path_rate, n_layers)
        ]  # stochastic depth decay rule
        # We are
        self.text_decoder = nn.ModuleList(
            [
                ResidualDecoderBlock(
                    d_model=transformer_width,
                    n_head=transformer_heads,
                    dropout=drop_rate,
                    drop_path=dpr[i],
                )
                for i in range(n_layers)
            ]
        )
        self.text_decoder_ln = nn.LayerNorm(transformer_width)

    def forward(self, text_feats, visual_feats):
        visual_feats = self.visual_encoder_to_decoder(visual_feats)
        text_feats = self.text_encoder_to_decoder(text_feats)
        # ! Shape
        # [L, N, T, C] -> [T * L, N, C]
        visual_feats = visual_feats.permute(2, 0, 1, 3).flatten(0, 1)
        # [N, L, C] -> [L, N, C]
        text_feats = text_feats.permute(1, 0, 2)
        for dec in self.text_decoder:
            text_feats = dec(text_feats, visual_feats)
        text_feats = self.text_decoder_ln(text_feats).permute(1, 0, 2)

        return text_feats


class MaskedVisualDecoder(nn.Module):
    def __init__(
        self,
        n_layers,
        transformer_width,
        vision_width,
        transformer_heads,
        patch_size,
        drop_path_rate=0.0,
    ):
        super().__init__()
        self.visual_encoder_to_decoder = nn.Sequential(
            nn.LayerNorm(vision_width), nn.Linear(vision_width, transformer_width)
        )
        self.text_encoder_to_decoder = nn.Sequential(
            nn.LayerNorm(transformer_width),
            nn.Linear(transformer_width, transformer_width),
        )
        dpr = [
            x.item() for x in torch.linspace(0, drop_path_rate, n_layers)
        ]  # stochastic depth decay rule
        # We are setting the d_model as transformer_width because we want the decoder to be small
        # Mayber later I will add a specific setting for this
        self.vision_decoder = nn.ModuleList(
            [
                ResidualDecoderBlock(
                    d_model=transformer_width,
                    n_head=transformer_heads,
                    drop_path=dpr[i],
                )
                for i in range(n_layers)
            ]
        )
        self.predictor = nn.Sequential(
            nn.LayerNorm(transformer_width),
            nn.Linear(transformer_width, transformer_width),
            nn.GELU(),
            nn.LayerNorm(transformer_width),
            nn.Linear(transformer_width, 3 * patch_size * patch_size),
        )

    def forward(self, text_feats, visual_feats):
        # Remove cls_token first
        visual_feats = self.visual_encoder_to_decoder(visual_feats[1:])
        text_feats = self.text_encoder_to_decoder(text_feats)
        # [L, N, T, C] -> [T * L, N, C]
        visual_feats = visual_feats.permute(2, 0, 1, 3).flatten(0, 1)
        # [N, L, C] -> [L, N, C]
        text_feats = text_feats.permute(1, 0, 2)
        for dec in self.vision_decoder:
            visual_feats = dec(visual_feats, text_feats)
        visual_feats = self.predictor(visual_feats).permute(1, 0, 2)  # [N, L, C]

        return visual_feats