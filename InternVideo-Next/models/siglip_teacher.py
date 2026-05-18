"""
# Adapted from https://huggingface.co/MILVLG/imp-v1-3b/blob/main/vision_encoder.py
"""

from typing import Optional, Tuple, Union, Dict
from dataclasses import dataclass
from functools import partial, reduce
from PIL import Image
import torch
import torch.utils.checkpoint
from torch import nn
import os
from transformers.image_processing_utils import BatchFeature, get_size_dict
from transformers.image_transforms import (
    convert_to_rgb,
    normalize,
    rescale,
    resize,
    to_channel_dimension_format,
)
from transformers.image_utils import (
    ChannelDimension,
    PILImageResampling,
    to_numpy_array,
)
from transformers.activations import ACT2FN
from transformers.modeling_outputs import BaseModelOutput, BaseModelOutputWithPooling
from transformers.modeling_utils import PreTrainedModel
from transformers import PretrainedConfig
from transformers.utils import ModelOutput

import math

# import sys
# import os
# root_path = "/mnt/petrelfs/linlang/Code/mar_d/dumt"
# sys.path.append(root_path)

from .transforms import *


class SigLipImageProcessor:
    def __init__(self, image_mean=(0.5, 0.5, 0.5), image_std=(0.5, 0.5, 0.5), size=(384, 384), crop_size: Dict[str, int] = None, resample=PILImageResampling.BICUBIC, rescale_factor=1 / 255, data_format=ChannelDimension.FIRST):
        crop_size = crop_size if crop_size is not None else {"height": size[0], "width": size[1]}
        crop_size = get_size_dict(crop_size, default_to_square=True, param_name="crop_size")

        self.image_mean = image_mean
        self.image_std = image_std
        self.size = size
        self.resample = resample
        self.rescale_factor = rescale_factor
        self.data_format = data_format
        self.crop_size = crop_size

    def preprocess(self, pixels):
        input_mean = [0.485, 0.456, 0.406]  # IMAGENET_DEFAULT_MEAN
        input_std = [0.229, 0.224, 0.225]   # IMAGENET_DEFAULT_STD
        
        denormalize = GroupDenormalize(input_mean, input_std)
        normalize = GroupNormalize(self.image_mean, self.image_std)
        # B C T H W
        pixels = normalize(denormalize(pixels.transpose(0, 1))).transpose(0, 1)
        
        return pixels


class SigLipVisionConfig(PretrainedConfig):
    model_type = "siglip_vision_model"

    def __init__(
        self,
        hidden_size=1152,
        image_mean=(0.5, 0.5, 0.5),
        intermediate_size=4304,
        num_hidden_layers=27,
        num_attention_heads=16,
        num_channels=3,
        image_size=384,
        patch_size=14,
        hidden_act="gelu_pytorch_tanh",
        layer_norm_eps=1e-6,
        attention_dropout=0.0,
        # For unmasked teacher
        clip_norm_type: str = None,
        depth: int = 27,
        clip_return_layer: int = 1,
        clip_return_interval: int = 1,
        clip_return_index: list = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.num_channels = num_channels
        self.patch_size = patch_size
        self.image_size = image_size
        self.attention_dropout = attention_dropout
        self.layer_norm_eps = layer_norm_eps
        self.hidden_act = hidden_act
        self.image_mean = image_mean
        # Unmasked Teacher
        print("curr norm type", clip_norm_type)
        self.clip_norm_type = clip_norm_type
        self.depth = num_hidden_layers
        self.clip_return_layer = clip_return_layer
        self.clip_return_interval = clip_return_interval
        self.clip_return_index = clip_return_index

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path: Union[str, os.PathLike], **kwargs) -> "PretrainedConfig":
        cls._set_token_in_kwargs(kwargs)

        config_dict, kwargs = cls.get_config_dict(pretrained_model_name_or_path, **kwargs)

        # get the vision config dict if we are loading from SigLipConfig
        if config_dict.get("model_type") == "siglip":
            config_dict = config_dict["vision_config"]

        if "model_type" in config_dict and hasattr(cls, "model_type") and config_dict["model_type"] != cls.model_type:
            print(f"You are using a model of type {config_dict['model_type']} to instantiate a model of type " f"{cls.model_type}. This is not supported for all configurations of models and can yield errors.")

        return cls.from_dict(config_dict, **kwargs)


@dataclass
# Copied from transformers.models.clip.modeling_clip.CLIPVisionModelOutput with CLIP->SigLip
class SigLipVisionModelOutput(ModelOutput):
    """
    Base class for vision model's outputs that also contains image embeddings of the pooling of the last hidden states.

    Args:
        image_embeds (`torch.FloatTensor` of shape `(batch_size, output_dim)` *optional* returned when model is initialized with `with_projection=True`):
            The image embeddings obtained by applying the projection layer to the pooler_output.
        last_hidden_state (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`):
            Sequence of hidden-states at the output of the last layer of the model.
        hidden_states (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, +
            one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.

            Hidden-states of the model at the output of each layer plus the optional initial embedding outputs.
        attentions (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
            sequence_length)`.

            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
            heads.
    """

    image_embeds: Optional[torch.FloatTensor] = None
    last_hidden_state: torch.FloatTensor = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None


class SigLipVisionEmbeddings(nn.Module):
    def __init__(self, config: SigLipVisionConfig):
        super().__init__()
        self.config = config
        self.embed_dim = config.hidden_size
        self.image_size = config.image_size
        self.patch_size = config.patch_size

        self.patch_embedding = nn.Conv2d(
            in_channels=config.num_channels,
            out_channels=self.embed_dim,
            kernel_size=self.patch_size,
            stride=self.patch_size,
            padding="valid",
        )

        self.num_patches = (self.image_size // self.patch_size) ** 2
        self.num_positions = self.num_patches
        self.grid = self.image_size // self.patch_size

        # TODO: How to enlarge to support flexible inference?
        self.position_embedding = nn.Embedding(self.num_positions, self.embed_dim) 

        self.register_buffer("position_ids", torch.arange(self.num_positions).expand((1, -1)), persistent=False)

    def forward(self, pixel_values: torch.FloatTensor) -> torch.Tensor:
        patch_embeds = self.patch_embedding(pixel_values)  # shape = [*, width, grid, grid]
        new_grid = patch_embeds.shape[-1]
        embeddings = patch_embeds.flatten(2).transpose(1, 2)
        pos_embed, position_ids = self.expand_pos_embed(self.position_embedding, new_grid)
        embeddings = embeddings + pos_embed(position_ids)
        return embeddings

    def expand_pos_embed(self, pos_embedding, new_grid):
        pos_embed = pos_embedding.weight
        pos_embed = pos_embed.reshape(1, self.grid, self.grid, self.embed_dim).permute(0, 3, 1, 2) # (*, C, H, W)
        pos_embed = torch.nn.functional.interpolate(
            pos_embed, size=(new_grid, new_grid), mode='bicubic', align_corners=False
        )
        pos_embed = pos_embed.permute(0, 2, 3, 1).reshape(new_grid ** 2, self.embed_dim)
        return nn.Embedding.from_pretrained(pos_embed, freeze=True).to('cuda'), torch.arange(new_grid ** 2).expand((1, -1)).to('cuda')
        

class SigLipAttention(nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.embed_dim = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.embed_dim // self.num_heads
        if self.head_dim * self.num_heads != self.embed_dim:
            raise ValueError(f"embed_dim must be divisible by num_heads (got `embed_dim`: {self.embed_dim} and `num_heads`:" f" {self.num_heads}).")
        self.scale = self.head_dim**-0.5
        self.dropout = config.attention_dropout

        self.k_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.v_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.q_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.out_proj = nn.Linear(self.embed_dim, self.embed_dim)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        """Input shape: Batch x Time x Channel"""

        batch_size, q_len, _ = hidden_states.size()

        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        query_states = query_states.view(batch_size, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(batch_size, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(batch_size, q_len, self.num_heads, self.head_dim).transpose(1, 2)

        k_v_seq_len = key_states.shape[-2]
        attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) * self.scale

        if attn_weights.size() != (batch_size, self.num_heads, q_len, k_v_seq_len):
            raise ValueError(f"Attention weights should be of size {(batch_size, self.num_heads, q_len, k_v_seq_len)}, but is" f" {attn_weights.size()}")

        if attention_mask is not None:
            if attention_mask.size() != (batch_size, 1, q_len, k_v_seq_len):
                raise ValueError(f"Attention mask should be of size {(batch_size, 1, q_len, k_v_seq_len)}, but is {attention_mask.size()}")
            attn_weights = attn_weights + attention_mask

        attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
        attn_weights = nn.functional.dropout(attn_weights, p=self.dropout, training=self.training)
        attn_output = torch.matmul(attn_weights, value_states)

        if attn_output.size() != (batch_size, self.num_heads, q_len, self.head_dim):
            raise ValueError(f"`attn_output` should be of size {(batch_size, self.num_heads, q_len, self.head_dim)}, but is" f" {attn_output.size()}")

        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(batch_size, q_len, self.embed_dim)

        attn_output = self.out_proj(attn_output)

        return attn_output, attn_weights


# Copied from transformers.models.clip.modeling_clip.CLIPMLP with CLIP->SigLip
class SigLipMLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.activation_fn = ACT2FN[config.hidden_act]
        self.fc1 = nn.Linear(config.hidden_size, config.intermediate_size)
        self.fc2 = nn.Linear(config.intermediate_size, config.hidden_size)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.fc1(hidden_states)
        hidden_states = self.activation_fn(hidden_states)
        hidden_states = self.fc2(hidden_states)
        return hidden_states


# Copied from transformers.models.clip.modeling_clip.CLIPEncoderLayer with CLIP->SigLip
class SigLipEncoderLayer(nn.Module):
    def __init__(self, config: SigLipVisionConfig):
        super().__init__()
        self.embed_dim = config.hidden_size
        self.self_attn = SigLipAttention(config)
        self.layer_norm1 = nn.LayerNorm(self.embed_dim, eps=config.layer_norm_eps)
        self.mlp = SigLipMLP(config)
        self.layer_norm2 = nn.LayerNorm(self.embed_dim, eps=config.layer_norm_eps)

    # Ignore copy
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor,
        output_attentions: Optional[bool] = False,
    ) -> Tuple[torch.FloatTensor]:
        """
        Args:
            hidden_states (`torch.FloatTensor`):
                Input to the layer of shape `(batch, seq_len, embed_dim)`.
            attention_mask (`torch.FloatTensor`):
                Attention mask of shape `(batch, 1, q_len, k_v_seq_len)` where padding elements are indicated by very large negative values.
            output_attentions (`bool`, *optional*, defaults to `False`):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under
                returned tensors for more detail.
        """
        residual = hidden_states

        hidden_states = self.layer_norm1(hidden_states)
        hidden_states, attn_weights = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            output_attentions=output_attentions,
        )
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.layer_norm2(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        outputs = (hidden_states,)

        if output_attentions:
            outputs += (attn_weights,)

        return outputs


class SigLipPreTrainedModel(PreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """

    config_class = SigLipVisionConfig
    base_model_prefix = "siglip"
    supports_gradient_checkpointing = True

    def _init_weights(self, module):
        """Initialize the weights"""
        pass


# Copied from transformers.models.clip.modeling_clip.CLIPEncoder with CLIP->SigLip
class SigLipEncoder(nn.Module):
    """
    Transformer encoder consisting of `config.num_hidden_layers` self attention layers. Each layer is a
    [`SigLipEncoderLayer`].

    Args:
        config: SigLipVisionConfig
    """

    def __init__(self, config: SigLipVisionConfig):
        super().__init__()
        self.config = config
        self.layers = nn.ModuleList([SigLipEncoderLayer(config) for _ in range(config.num_hidden_layers)])
        self.gradient_checkpointing = False
        
        self.return_index = []
        if config.clip_return_index:
            self.return_index = config.clip_return_index
        else:
            for i in range(config.clip_return_layer):
                self.return_index.append(config.depth - int(i * config.clip_return_interval) - 1)

    # Ignore copy
    def forward(
        self,
        inputs_embeds,
        attention_mask: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None, # modified to only output hidden states of required
        return_dict: Optional[bool] = None,
    ):
        r"""
        Args:
            inputs_embeds (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`):
                Optionally, instead of passing `input_ids` you can choose to directly pass an embedded representation.
                This is useful if you want more control over how to convert `input_ids` indices into associated vectors
                than the model's internal embedding lookup matrix.
            attention_mask (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
                Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:

                - 1 for tokens that are **not masked**,
                - 0 for tokens that are **masked**.

                [What are attention masks?](../glossary#attention-mask)
            output_attentions (`bool`, *optional*):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under
                returned tensors for more detail.
            output_hidden_states (`bool`, *optional*):
                Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors
                for more detail. See the notes above for further details.
            return_dict (`bool`, *optional*):
                Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.
        """
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        
        assert output_hidden_states == True

        encoder_states = () if output_hidden_states else None
        all_attentions = () if output_attentions else None

        hidden_states = inputs_embeds
        for layer_id, encoder_layer in enumerate(self.layers):
            if output_hidden_states and layer_id in self.return_index:
                encoder_states = encoder_states + (hidden_states,)
            if self.gradient_checkpointing and self.training:
                layer_outputs = self._gradient_checkpointing_func(
                    encoder_layer.__call__,
                    hidden_states,
                    attention_mask,
                    output_attentions,
                )
            else:
                layer_outputs = encoder_layer(
                    hidden_states,
                    attention_mask,
                    output_attentions=output_attentions,
                )

            hidden_states = layer_outputs[0]

            if output_attentions:
                all_attentions = all_attentions + (layer_outputs[1],)

        return tuple(v for v in [hidden_states, encoder_states, all_attentions] if v is not None)


class SigLipVisionTransformer(nn.Module):
    def __init__(self, config: SigLipVisionConfig):
        super().__init__()
        self.config = config
        embed_dim = config.hidden_size

        self.embeddings = SigLipVisionEmbeddings(config)
        self.encoder = SigLipEncoder(config)
        self.post_layernorm = nn.LayerNorm(embed_dim, eps=config.layer_norm_eps)
        self.head = SigLipMultiheadAttentionPoolingHead(config)

    def forward(
        self,
        pixel_values,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        mask_ratio: Optional[float] = None,
    ) -> Union[Tuple, BaseModelOutputWithPooling]:
        r"""
        Returns:

        """
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        x = self.embeddings(pixel_values) # B, L, C
        B, L, C = x.shape
        assert int(math.sqrt(L)) ** 2 == L

        encoder_outputs = self.encoder(
            inputs_embeds=x,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        last_hidden_state = encoder_outputs[0]
        last_hidden_state = self.post_layernorm(last_hidden_state)
        x, attn = self.head(last_hidden_state)
        
        encoder_states = encoder_outputs[1]
        
        if self.encoder.return_index is not None:
            # assert return_dict is False
            dir_teacher_feats = torch.stack(encoder_states)
            # Normalization for features
            if self.config.clip_norm_type == 'l2':
                dir_teacher_feats = dir_teacher_feats / dir_teacher_feats.norm(dim=-1, keepdim=True)
                x = x / x.norm(dim=-1, keepdim=True)
            else:
                raise NotImplementedError
            # return dir_teacher_feats, x, attn, mask
            if output_attentions:
                return dir_teacher_feats, x, attn, None, encoder_outputs[2]
            return dir_teacher_feats, x, attn, None
        else:
            raise NotImplementedError


class SigLipMultiheadAttentionPoolingHead(nn.Module):
    """Multihead Attention Pooling."""

    def __init__(self, config: SigLipVisionConfig):
        super().__init__()

        self.probe = nn.Parameter(torch.randn(1, 1, config.hidden_size))
        self.attention = torch.nn.MultiheadAttention(config.hidden_size, config.num_attention_heads, batch_first=True)
        self.layernorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.mlp = SigLipMLP(config)

    def forward(self, hidden_state):
        batch_size = hidden_state.shape[0]
        probe = self.probe.repeat(batch_size, 1, 1)

        hidden_state, attn = self.attention(probe, hidden_state, hidden_state)

        residual = hidden_state
        hidden_state = self.layernorm(hidden_state)
        hidden_state = residual + self.mlp(hidden_state)
        # if return_attn:
        return hidden_state[:, 0], attn[:, 0]
        # return hidden_state[:, 0]


class SigLipVisionModel(SigLipPreTrainedModel):
    config_class = SigLipVisionConfig
    main_input_name = "pixel_values"
    _no_split_modules = ["SigLipEncoderLayer"]

    def __init__(self, config: SigLipVisionConfig):
        super().__init__(config)

        self.vision_model = SigLipVisionTransformer(config)

        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self) -> nn.Module:
        return self.vision_model.embeddings.patch_embedding

    def forward(
        self,
        pixel_values,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        mask_ratio: Optional[float] = None,
    ) -> Union[Tuple, BaseModelOutputWithPooling]:
        r"""
        Returns:

        Examples:

        ```python
        >>> from PIL import Image
        >>> import requests
        >>> from transformers import AutoProcessor, SigLipVisionModel

        >>> model = SigLipVisionModel.from_pretrained("google/siglip-base-patch16-224")
        >>> processor = AutoProcessor.from_pretrained("google/siglip-base-patch16-224")

        >>> url = "http://images.cocodataset.org/val2017/000000039769.jpg"
        >>> image = Image.open(requests.get(url, stream=True).raw)

        >>> inputs = processor(images=image, return_tensors="pt")

        >>> outputs = model(**inputs)
        >>> last_hidden_state = outputs.last_hidden_state
        >>> pooled_output = outputs.pooler_output  # pooled features
        ```"""
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        return self.vision_model(
            pixel_values=pixel_values,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            mask_ratio=mask_ratio,
        )


class SigLipVisionTower(nn.Module):
    def __init__(self, is_siglip2=False, res=384, size='so', **kwargs):
        super().__init__()

        self.is_loaded = False

        if res == 384:
            if size == '1b':
                self.vision_tower_name = '...' # {pre downloaded path}
            elif is_siglip2 is False:
                self.vision_tower_name = '...'
            else:
                self.vision_tower_name = '...'
        else:
            if size == 'so':
                self.vision_tower_name = '...'
            else:
                self.vision_tower_name = '...'

        self.config = SigLipVisionConfig(image_size=res, **kwargs)

        self.image_processor = SigLipImageProcessor(size=(res, res))

        self.load_model()

    def load_model(self, device_map=None):
        if self.is_loaded:
            return

        self.vision_tower = SigLipVisionModel.from_pretrained(self.vision_tower_name, device_map=device_map, config=self.config)

        self.vision_tower.requires_grad_(False)

        self.is_loaded = True

    def forward(self, images, mask_ratio = None, return_mask=False, return_attn=False):
        if type(images) is list:
            images = torch.stack(images, dim=2) # Expect to be: B C T H W
        images = self.image_processor.preprocess(images)

        if return_attn:
            dir_teacher_feats, x, attn, mask, attentions = self.vision_tower(images.permute(0, 2, 1, 3, 4).flatten(0, 1).to(device=self.device, dtype=self.dtype), output_hidden_states=True, mask_ratio=mask_ratio, output_attentions=True)
            return dir_teacher_feats, x, attn, attentions
        
        dir_teacher_feats, x, attn, mask = self.vision_tower(images.permute(0, 2, 1, 3, 4).flatten(0, 1).to(device=self.device, dtype=self.dtype), output_hidden_states=True, mask_ratio=mask_ratio)
        
        if return_mask:
            return dir_teacher_feats, x, attn, mask
        else:
            return dir_teacher_feats, x, attn

    @property
    def dummy_feature(self):
        return torch.zeros(1, self.hidden_size, device=self.device, dtype=self.dtype)

    @property
    def dtype(self):
        for p in self.vision_tower.parameters():
            return p.dtype

    @property
    def device(self):
        for p in self.vision_tower.parameters():
            return p.device

    @property
    def hidden_size(self):
        return self.config.hidden_size

    @property
    def num_patches(self):
        return (self.config.image_size // self.config.patch_size) ** 2

    @property
    def num_patches_per_side(self):
        return self.config.image_size // self.config.patch_size

    @property
    def image_size(self):
        return self.config.image_size

def teacher_siglip_400M_once4all_mm_umt(
        clip_norm_type='l2',
        return_attn=True,
        clip_return_layer=1,
        clip_return_interval=1,
        clip_return_index=None,
    ):
    model = SigLipVisionTower(
        clip_norm_type=clip_norm_type,
        return_attn=return_attn,
        clip_return_layer=clip_return_layer,
        clip_return_interval=clip_return_interval,
        clip_return_index=clip_return_index,
    )
    print("Loaded teacher model Siglip. Successed.")
    return model

def teacher_siglip2_400M_once4all_mm_umt(
        clip_norm_type='l2',
        return_attn=True,
        clip_return_layer=1,
        clip_return_interval=1,
        clip_return_index=None,
    ):
    model = SigLipVisionTower(
        clip_norm_type=clip_norm_type,
        return_attn=return_attn,
        clip_return_layer=clip_return_layer,
        clip_return_interval=clip_return_interval,
        clip_return_index=clip_return_index,
        is_siglip2=True,
    )
    print("Loaded teacher model Siglip. Successed.")
    return model

def teacher_siglip2_1b_once4all_mm_umt_res256(
    clip_norm_type='l2',
    return_attn=True,
    clip_return_layer=1,
    clip_return_interval=1,
    clip_return_index=None,
):
    model = SigLipVisionTower(
        clip_norm_type=clip_norm_type,
        return_attn=return_attn,
        clip_return_layer=clip_return_layer,
        clip_return_interval=clip_return_interval,
        clip_return_index=clip_return_index,
        is_siglip2=True,
        res=256,
        size='1b',
        intermediate_size=6144,
        hidden_size=1536,
        num_attention_heads=16,
        num_hidden_layers=40,
        patch_size=16,
    )
    print("Loaded teacher model Siglip. Successed.")
    return model

def teacher_siglip2_1b_once4all_mm_umt_res384(
    clip_norm_type='l2',
    return_attn=True,
    clip_return_layer=1,
    clip_return_interval=1,
    clip_return_index=None,
):
    model = SigLipVisionTower(
        clip_norm_type=clip_norm_type,
        return_attn=return_attn,
        clip_return_layer=clip_return_layer,
        clip_return_interval=clip_return_interval,
        clip_return_index=clip_return_index,
        is_siglip2=True,
        res=384,
        size='1b',
        intermediate_size=6144,
        hidden_size=1536,
        num_attention_heads=16,
        num_hidden_layers=40,
        patch_size=16,
    )
    print("Loaded teacher model Siglip. Successed.")
    return model

def teacher_siglip2_400M_once4all_mm_umt_res224(
        clip_norm_type='l2',
        return_attn=True,
        clip_return_layer=1,
        clip_return_interval=1,
        clip_return_index=None,
    ):
    model = SigLipVisionTower(
        clip_norm_type=clip_norm_type,
        return_attn=return_attn,
        clip_return_layer=clip_return_layer,
        clip_return_interval=clip_return_interval,
        clip_return_index=clip_return_index,
        is_siglip2=True,
        res=224,
    )
    print("Loaded teacher model Siglip. Successed.")
    return model