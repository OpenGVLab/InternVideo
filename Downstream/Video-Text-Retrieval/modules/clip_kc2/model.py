from collections import OrderedDict
from typing import Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from . import evl_utils
from .evl_utils import TransformerDecoder_uniformer_diff_conv_balance
from einops import rearrange
from ipdb import set_trace
from copy import deepcopy
from .clip_decoders import CaptionDecoder


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


class CLIP(nn.Module):
    def __init__(self,
                 embed_dim: int,
                 # vision
                 image_resolution: int,
                 vision_layers: Union[Tuple[int, int, int, int], int],
                 vision_width: int,
                 vision_patch_size: int,
                 # text
                 context_length: int,
                 vocab_size: int,
                 transformer_width: int,
                 transformer_heads: int,
                 transformer_layers: int,
                 # evl
                 n_layers=4, n_dim=768, n_head=12, mlp_factor=4.0, drop_path_rate=0.,
                 mlp_dropout=[0.5, 0.5, 0.5, 0.5], cls_dropout=0.5, t_size=8, spatial_size=14,
                 use_t_conv=True, use_image_attnmap=True, use_t_pos_embed=True,
                 backbone='vit_2plus1d_dw_bias_b16',
                 uni_layer=0,
                 uni_type='2d',
                 add_ffn=False,
                 t_conv_type='3d',
                 pre_prompt=False,
                 balance=0.,
                 after_me=True, 
                 before_me=False, 
                 me_type='stm', 
                 me_reduction=4,
                 init_zero=True,
                 use_capdecoder=False,
                ):
        super().__init__()
        
        # All assertions is for adhoc clip_kc and should be removed
        # assert vision_layers == 12, vision_layers
        assert image_resolution == 224, image_resolution
        # assert vision_patch_size == 32, vision_patch_size
        assert vision_width == n_dim, (vision_width, n_dim)

        self.vision_width = n_dim

        self.context_length = context_length

        vision_heads = vision_width // 64
        self.visual = evl_utils.__dict__[backbone](pretrained=False, init_zero=init_zero, num_frames=t_size, t_size=t_size)
        self.evl = TransformerDecoder_uniformer_diff_conv_balance(
            n_layers=n_layers, n_dim=n_dim, n_head=n_head, 
            mlp_factor=mlp_factor, drop_path_rate=drop_path_rate,
            mlp_dropout=mlp_dropout, cls_dropout=cls_dropout, t_size=t_size, 
            use_t_conv=use_t_conv, use_t_pos_embed=use_t_pos_embed,
            uni_layer=uni_layer, uni_type=uni_type, add_ffn=add_ffn, t_conv_type=t_conv_type, 
            pre_prompt=pre_prompt, balance=balance,
            after_me=after_me, before_me=before_me, 
            me_type=me_type, me_reduction=me_reduction,
            init_zero=init_zero,
        )
        self.visual_ln_post = nn.LayerNorm(n_dim)
        scale =  n_dim ** -0.5
        self.visual_proj = nn.Parameter(scale * torch.randn(n_dim, embed_dim))
        self.return_qk = use_image_attnmap
        self.return_num = n_layers

        self.transformer = Transformer(
            width=transformer_width,
            layers=transformer_layers,
            heads=transformer_heads,
            attn_mask=self.build_attention_mask(),
        )

        self.vocab_size = vocab_size
        self.token_embedding = nn.Embedding(vocab_size, transformer_width)
        self.positional_embedding = nn.Parameter(torch.empty(self.context_length, transformer_width))
        self.ln_final = LayerNorm(transformer_width)

        self.text_projection = nn.Parameter(torch.empty(transformer_width, embed_dim))
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

        self.embed_dim = embed_dim

        # We seperate the mask embedding to load pretrained model
        self.text_mask_embedding = nn.Parameter(torch.empty(1, 1, transformer_width))

        # # To keep the num_embeddings unchanged, we add this to embedded text
        # self.eot_token_embedding = nn.Parameter(torch.empty(1, transformer_width))
        
        ##################### try cap decoder ###############
        self.use_capdecoder = use_capdecoder
        if self.use_capdecoder:
            self.caption_decoder = CaptionDecoder(
                n_layers=6,
                transformer_width=transformer_width,
                vision_width=vision_width,
                transformer_heads=transformer_heads,
                vocab_size=vocab_size,
            )
            ## we do not use logits, just text feats ##
            self.caption_decoder.predictor = nn.Identity()
        #####################################################


        self.initialize_parameters()

    def initialize_parameters(self):
        nn.init.normal_(self.token_embedding.weight, std=0.02)
        nn.init.normal_(self.positional_embedding, std=0.01)
        nn.init.normal_(self.text_mask_embedding, std=0.02)
        # nn.init.constant_(self.eot_token_embedding, 0.0)

        proj_std = (self.transformer.width ** -0.5) * ((2 * self.transformer.layers) ** -0.5)
        attn_std = self.transformer.width ** -0.5
        fc_std = (2 * self.transformer.width) ** -0.5
        for block in self.transformer.resblocks:
            nn.init.normal_(block.attn.in_proj_weight, std=attn_std)
            nn.init.normal_(block.attn.out_proj.weight, std=proj_std)
            nn.init.normal_(block.mlp.c_fc.weight, std=fc_std)
            nn.init.normal_(block.mlp.c_proj.weight, std=proj_std)

        if self.text_projection is not None:
            nn.init.normal_(self.text_projection, std=self.transformer.width ** -0.5)
        
        nn.init.constant_(self.visual_ln_post.weight, 1.0)
        nn.init.constant_(self.visual_ln_post.bias, 0.0)

    def build_attention_mask(self):
        # lazily create causal attention mask, with full attention between the vision tokens
        # pytorch uses additive attention mask; fill with -inf
        mask = torch.empty(self.context_length, self.context_length)
        mask.fill_(float("-inf"))
        mask.triu_(1)  # zero out the lower diagonal
        return mask

    @property
    def dtype(self):
        return self.visual.conv1.weight.dtype

    def encode_video(self, video, return_all_feats=False, masked_indices=None, mode="video"):
        # if len(video.size()) == 4: #[bs * T, C, H, W]
            #set_trace()
            # frames = 12
            # video = rearrange(video, '(b t) c h w -> b t c h w', b=int(video.size(0)/frames), t=frames)
            # video = rearrange(video, 'b t c h w -> b c t h w')
        
        #set_trace()
        # video: [N, C, T, H, W]
        features = self.visual(video, return_num=self.return_num, masked_indices=masked_indices, mode=mode)
        x = self.visual_ln_post(self.evl(features))
        if self.visual_proj is not None:
            x = x @ self.visual_proj

        if return_all_feats:
            return x, features[-1]
            # return x, features[-1], features[-1][0]  # [N, T, C], [L, N, T, C], [N, T, C]

        return x

    def encode_text(self, text, masked_indices=None, return_all_feats=False):
        # assert (text.max(dim=-1)[0] + 1 == self.token_embedding.num_embeddings).all(), \
        #     "The last token of each sentence should be eot_token, check the input"

        x = self.token_embedding(text).type(self.dtype)  # [batch_size, n_ctx, d_model]
        # x[torch.arange(x.shape[0]), text.argmax(dim=-1)] += self.eot_token_embedding

        if masked_indices is not None:
            x[masked_indices] = self.text_mask_embedding

        x = x + self.positional_embedding.type(self.dtype)
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x).type(self.dtype)

        # x.shape = [batch_size, n_ctx, transformer.width]
        # take features from the eot embedding (eot_token is the highest number in each sequence)
        feats = x[torch.arange(x.shape[0]), text.argmax(dim=-1)]
        if self.text_projection is not None:
            feats = feats @ self.text_projection

        if return_all_feats:
            return feats, x

        return feats

    def forward(self, video, text):
        video_features = self.encode_video(video)
        text_features = self.encode_text(text)

        # normalized features
        video_features = video_features / video_features.norm(dim=1, keepdim=True)
        text_features = text_features / text_features.norm(dim=1, keepdim=True)

        # cosine similarity as logits
        logit_scale = self.logit_scale.exp()
        logits_per_video= logit_scale * video_features @ text_features.t()
        logits_per_text = logits_per_video.t()

        # shape = [global_batch_size, global_batch_size]
        return logits_per_video, logits_per_text


def convert_weights(model: nn.Module):
    """Convert applicable model parameters to fp16"""

    def _convert_weights_to_fp16(l):
        if isinstance(l, (nn.Conv1d, nn.Conv2d, nn.Linear)):
            l.weight.data = l.weight.data.half()
            if l.bias is not None:
                l.bias.data = l.bias.data.half()

        if isinstance(l, nn.MultiheadAttention):
            for attr in [*[f"{s}_proj_weight" for s in ["in", "q", "k", "v"]], "in_proj_bias", "bias_k", "bias_v"]:
                tensor = getattr(l, attr)
                if tensor is not None:
                    tensor.data = tensor.data.half()

        for name in ["text_projection", "proj"]:
            if hasattr(l, name) and not isinstance(l, TransformerDecoder_uniformer_diff_conv_balance):
                attr = getattr(l, name)
                if attr is not None:
                    attr.data = attr.data.half()

    model.apply(_convert_weights_to_fp16)

def interpolate_temporal_pos_embed(pos_embed, T):
    # [1, t, d] -> [1, d, t]
    pos_embed = pos_embed.transpose(-2, -1)
    # [1, d, t] -> [1, d, T]
    pos_embed = F.interpolate(pos_embed, size=(T), mode='linear')
    # [1, d, T] -> [1, T, d]
    return pos_embed.transpose(-2, -1)

def interploate_rpb(rpb, T):
    t1 = T * 2 - 1
    rpb = rpb.transpose(0, 1).unsqueeze(0)
    rpb = F.interpolate(rpb, size=(t1), mode='linear')
    return rpb.squeeze(0).transpose(0, 1)
    
def build_model(
        state_dict: dict,
        # evl
        n_layers=4, n_dim=768, n_head=12, mlp_factor=4.0, drop_path_rate=0.,
        mlp_dropout=[0.5, 0.5, 0.5, 0.5], cls_dropout=0.5, t_size=8, spatial_size=14,
        use_t_conv=True, use_image_attnmap=True, use_t_pos_embed=True, no_pretrain=False,
        init_zero=True, mergeclip=False, mergeweight=0.5, use_capdecoder=False, clip_state_dict=None, 
    ):
    vit = "visual.proj" in state_dict or "visual.positional_embedding" in state_dict

    if "visual.proj" in state_dict:
        state_dict["visual_proj"] = state_dict["visual.proj"]
        state_dict["visual_ln_post.weight"] = state_dict["visual.ln_post.weight"]
        state_dict["visual_ln_post.bias"] = state_dict["visual.ln_post.bias"]
        del state_dict["visual.proj"], state_dict["visual.ln_post.weight"], state_dict["visual.ln_post.bias"]
    # new_state_dict = OrderedDict()
    # for k, v in state_dict.items():
    #     if k.startswith("backbone."):
    #         k = k.replace("backbone.", "visual.")
        
    #     new_state_dict[k] = v
    
    # state_dict = new_state_dict
    if vit:
        vision_width = state_dict["visual.conv1.weight"].shape[0]
        vision_layers = len([k for k in state_dict.keys() if k.startswith("visual.") and k.endswith(".attn.in_proj_weight")])
        vision_patch_size = state_dict["visual.conv1.weight"].shape[-1]
        grid_size = round((state_dict["visual.positional_embedding"].shape[0] - 1) ** 0.5)
        image_resolution = vision_patch_size * grid_size
    else:
        counts: list = [len(set(k.split(".")[2] for k in state_dict if k.startswith(f"visual.layer{b}"))) for b in [1, 2, 3, 4]]
        vision_layers = tuple(counts)
        vision_width = state_dict["visual.layer1.0.conv1.weight"].shape[0]
        output_width = round((state_dict["visual.attnpool.positional_embedding"].shape[0] - 1) ** 0.5)
        vision_patch_size = None
        assert output_width ** 2 + 1 == state_dict["visual.attnpool.positional_embedding"].shape[0]
        image_resolution = output_width * 32

    # embed_dim = 512
    # context_length = 77
    # vocab_size = 49408
    # transformer_width = 512
    # transformer_layers = 12
    embed_dim = state_dict["text_projection"].shape[1]
    context_length = state_dict["positional_embedding"].shape[0]
    vocab_size = state_dict["token_embedding.weight"].shape[0]
    transformer_width = state_dict["ln_final.weight"].shape[0]
    transformer_heads = transformer_width // 64
    transformer_layers = len(set(k.split(".")[2] for k in state_dict if k.startswith(f"transformer.resblocks")))

    vision_width = state_dict["visual_proj"].shape[0]
    n_dim = vision_width
    if vision_width == 768:
        backbone = "vit_2plus1d_dw_bias_b16"
        n_head = 12
    elif vision_width == 1024:
        backbone = "vit_2plus1d_dw_bias_l14"
        n_head = 16
    else:
        raise NotImplementedError

    model = CLIP(
        embed_dim,
        image_resolution, vision_layers, vision_width, vision_patch_size,
        context_length, vocab_size, transformer_width, transformer_heads, transformer_layers,
        n_layers=n_layers, n_dim=n_dim, n_head=n_head, mlp_factor=mlp_factor, drop_path_rate=drop_path_rate,
        mlp_dropout=mlp_dropout, cls_dropout=cls_dropout, t_size=t_size, spatial_size=spatial_size,
        use_t_conv=use_t_conv, use_image_attnmap=use_image_attnmap, use_t_pos_embed=use_t_pos_embed, backbone=backbone,
        init_zero=init_zero, use_capdecoder=use_capdecoder,
    )

    for key in ["input_resolution", "context_length", "vocab_size"]:
        if key in state_dict:
            del state_dict[key]

    ###### convert_weights(model)#####
    ##### interpolate pos embedding ######
    temporal_key = 'visual.temporal_positional_embedding'
    temporal_key2 = 'evl.pemb_t'
    
    if temporal_key in state_dict and t_size != state_dict[temporal_key].size(1):
        state_dict[temporal_key] = interpolate_temporal_pos_embed(state_dict[temporal_key], t_size)
        state_dict[temporal_key2] = interpolate_temporal_pos_embed(state_dict[temporal_key2], t_size)
        for kk, vv in state_dict.items():
            if 'rpb_t' in kk:
                size_old = state_dict[kk].shape
                state_dict[kk] = interploate_rpb(vv, t_size)
                size_new = state_dict[kk].shape
                print('Interpolating' ,kk, size_old, '-->', size_new)
    # set_trace()
    if mergeclip:
        assert 0.0 <= mergeweight <= 1.0
        assert clip_state_dict is not None
        clip_sd = {k: v.cpu() for k, v in clip_state_dict.items()}
        new_sd = deepcopy(state_dict)
        for k in new_sd:
            if k not in clip_sd:
                continue
            if any(x in k for x in clip_sd.keys()):
                print('merging: ', k)
                new_sd[k] = clip_sd[k] * mergeweight + state_dict[k] * (1.0 - mergeweight)
                
                ############## only merge the clip text features, this is for ActivityNet ###########
                # if 'visual' in k: 
                #     new_sd[k] = clip_sd[k]
                # else:
                #     new_sd[k] = clip_sd[k] * mergeweight + state_dict[k] * (1.0 - mergeweight)
                ################################################################################

                ############## only merge the clip visual features, this is for MSVD ###########
                # if 'visual' in k: 
                #     new_sd[k] = clip_sd[k] * mergeweight + state_dict[k] * (1.0 - mergeweight)
                # else:
                #     new_sd[k] = clip_sd[k]
                ################################################################################
        state_dict = new_sd
    

    # strict=False, for parameters of decoder
    # assert False, (len(model.state_dict()), len(state_dict))
    if not no_pretrain:
        msg = model.load_state_dict(state_dict, strict=False)
        print(msg)
    return model.eval()
