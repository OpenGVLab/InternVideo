import os
import logging
from collections import OrderedDict
from pkg_resources import packaging
from .simple_tokenizer import SimpleTokenizer as _Tokenizer

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

logger = logging.getLogger(__name__)


OPENCLIP_MODEL_PATH = 'https://huggingface.co/laion/CLIP-ViT-L-14-DataComp.XL-s13B-b90K'
_MODELS = {
    "ViT-L/14": "https://huggingface.co/laion/CLIP-ViT-L-14-DataComp.XL-s13B-b90K/vit_l14_text.pth",
    "CLIP-ViT-L/14": "https://huggingface.co/openai/clip-vit-large-patch14-336/vit_l14_text.pth",
    "CLIP-ViT-B/16": "https://huggingface.co/openai/clip-vit-base-patch16/vit_b16_text.pth",
}


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
        self.mlp = nn.Sequential(OrderedDict([
            ("c_fc", nn.Linear(d_model, d_model * 4)),
            ("gelu", QuickGELU()),
            ("c_proj", nn.Linear(d_model * 4, d_model))
        ]))
        self.ln_2 = LayerNorm(d_model)
        self.attn_mask = attn_mask

    def attention(self, x: torch.Tensor):
        self.attn_mask = self.attn_mask.to(dtype=x.dtype, device=x.device) if self.attn_mask is not None else None
        return self.attn(x, x, x, need_weights=False, attn_mask=self.attn_mask)[0]

    def forward(self, x: torch.Tensor):
        x = x + self.attention(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


class Transformer(nn.Module):
    def __init__(self, width: int, layers: int, heads: int, attn_mask: torch.Tensor = None):
        super().__init__()
        self.width = width
        self.layers = layers
        self.resblocks = nn.Sequential(*[ResidualAttentionBlock(width, heads, attn_mask) for _ in range(layers)])

    def forward(self, x: torch.Tensor):
        return self.resblocks(x)


class CLIP_TEXT(nn.Module):
    def __init__(
            self,
            embed_dim: int,
            context_length: int,
            vocab_size: int,
            transformer_width: int,
            transformer_heads: int,
            transformer_layers: int
        ):
        super().__init__()

        self.context_length = context_length
        self._tokenizer = _Tokenizer()

        self.transformer = Transformer(
            width=transformer_width,
            layers=transformer_layers,
            heads=transformer_heads,
            attn_mask=self.build_attention_mask()
        )

        self.vocab_size = vocab_size
        self.token_embedding = nn.Embedding(vocab_size, transformer_width)
        self.positional_embedding = nn.Parameter(torch.empty(self.context_length, transformer_width))
        self.ln_final = LayerNorm(transformer_width)

        self.text_projection = nn.Parameter(torch.empty(transformer_width, embed_dim))

    def build_attention_mask(self):
        # lazily create causal attention mask, with full attention between the vision tokens
        # pytorch uses additive attention mask; fill with -inf
        mask = torch.empty(self.context_length, self.context_length)
        mask.fill_(float("-inf"))
        mask.triu_(1)  # zero out the lower diagonal
        return mask

    def tokenize(self, texts, context_length=77, truncate=True):
        """
        Returns the tokenized representation of given input string(s)
        Parameters
        ----------
        texts : Union[str, List[str]]
            An input string or a list of input strings to tokenize
        context_length : int
            The context length to use; all CLIP models use 77 as the context length
        truncate: bool
            Whether to truncate the text in case its encoding is longer than the context length
        Returns
        -------
        A two-dimensional tensor containing the resulting tokens, shape = [number of input strings, context_length].
        We return LongTensor when torch version is <1.8.0, since older index_select requires indices to be long.
        """
        if isinstance(texts, str):
            texts = [texts]

        sot_token = self._tokenizer.encoder["<|startoftext|>"]
        eot_token = self._tokenizer.encoder["<|endoftext|>"]
        all_tokens = [[sot_token] + self._tokenizer.encode(text) + [eot_token] for text in texts]
        if packaging.version.parse(torch.__version__) < packaging.version.parse("1.8.0"):
            result = torch.zeros(len(all_tokens), context_length, dtype=torch.long)
        else:
            result = torch.zeros(len(all_tokens), context_length, dtype=torch.int)

        for i, tokens in enumerate(all_tokens):
            if len(tokens) > context_length:
                if truncate:
                    tokens = tokens[:context_length]
                    tokens[-1] = eot_token
                else:
                    raise RuntimeError(f"Input {texts[i]} is too long for context length {context_length}")
            result[i, :len(tokens)] = torch.tensor(tokens)

        return result

    def forward(self, texts):
        text = self.tokenize(texts).cuda()
        x = self.token_embedding(text)  # [batch_size, n_ctx, d_model]

        x = x + self.positional_embedding
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x)

        # x.shape = [batch_size, n_ctx, transformer.width]
        # take features from the eot embedding (eot_token is the highest number in each sequence)
        x = x[torch.arange(x.shape[0]), text.argmax(dim=-1)] @ self.text_projection

        return x


def clip_text_b16(
    embed_dim=512,
    context_length=77,
    vocab_size=49408,
    transformer_width=512,
    transformer_heads=8,
    transformer_layers=12,
):
    model = CLIP_TEXT(
        embed_dim,
        context_length,
        vocab_size,
        transformer_width,
        transformer_heads,
        transformer_layers
    )
    pretrained = _MODELS["ViT-B/16"]
    logger.info(f"Load pretrained weights from {pretrained}")
    state_dict = torch.load(pretrained, map_location='cpu')
    model.load_state_dict(state_dict, strict=False)
    return model.eval()


def clip_text_l14(
    embed_dim=768,
    context_length=77,
    vocab_size=49408,
    transformer_width=768,
    transformer_heads=12,
    transformer_layers=12,
):
    model = CLIP_TEXT(
        embed_dim,
        context_length,
        vocab_size,
        transformer_width,
        transformer_heads,
        transformer_layers
    )
    pretrained = _MODELS["ViT-L/14"]
    logger.info(f"Load pretrained weights from {pretrained}")
    state_dict = torch.load(pretrained, map_location='cpu')
    model.load_state_dict(state_dict, strict=False)
    return model.eval()


def clip_text_l14_336(
    embed_dim=768,
    context_length=77,
    vocab_size=49408,
    transformer_width=768,
    transformer_heads=12,
    transformer_layers=12,
):
    model = CLIP_TEXT(
        embed_dim,
        context_length,
        vocab_size,
        transformer_width,
        transformer_heads,
        transformer_layers
    )
    pretrained = _MODELS["ViT-L/14_336"]
    logger.info(f"Load pretrained weights from {pretrained}")
    state_dict = torch.load(pretrained, map_location='cpu')
    model.load_state_dict(state_dict, strict=False)
    return model.eval()


def build_clip(config):
    model_cls = config.text_encoder.clip_teacher
    model = eval(model_cls)()
    return model


if __name__ == '__main__':
    import time
    from fvcore.nn import FlopCountAnalysis
    from fvcore.nn import flop_count_table
    import numpy as np

    seed = 4217
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    num_frames = 4
    
    config = {
        'text_encoder':
            {
            'clip_teacher': 'clip_text_b16',
        }
    }
    from easydict import EasyDict
    model = build_clip(EasyDict(config))
    
    output = model('This is a dog')
    print(output.shape)