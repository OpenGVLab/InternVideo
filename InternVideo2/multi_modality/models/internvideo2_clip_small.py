import logging
import os
import json

import torch
from torch import nn
import numpy as np
from PIL import Image
import torchvision.transforms as transforms
from torchvision.transforms import InterpolationMode

from .backbones.internvideo2 import InternVideo2, TextTransformer, ClipTokenizer
from .criterions import VTC_VTM_Loss

logger = logging.getLogger(__name__)


class InternVideo2_CLIP_small(nn.Module):
    def __init__(self, config, tokenizer=None, is_pretrain=True):
        super().__init__()

        self.config = config
        self.tokenizer = tokenizer
        self.is_pretrain = is_pretrain

        # create modules.
        text_encoder_cfg = json.load(
            open(os.path.join(
                "./models/backbones/internvideo2/mobileclip/configs/" + \
                f"{self.config.model.text_encoder.name}.json"))
        )
        if tokenizer is None:
            self.tokenizer = ClipTokenizer(text_encoder_cfg)
        self.vision_encoder = self.build_vision_encoder()
        self.vision_align = nn.Sequential(
            nn.LayerNorm(self.config.model.vision_encoder.clip_embed_dim),
            nn.Linear(
                self.config.model.vision_encoder.clip_embed_dim, 
                self.config.model.vision_encoder.align_dim
            ),
        )
        self.text_encoder = self.build_text_encoder(cfg=text_encoder_cfg['text_cfg'], projection_dim=text_encoder_cfg["embed_dim"])
        # adopt 1 / 100. as in ViCLIP
        self.temp = nn.parameter.Parameter(torch.ones([]) * config.model.temp)
        self.temp_min = config.model.temp_min
        
        # freeze model
        if self.config.model.freeze_vision:
            for name, p in self.vision_encoder.named_parameters():
                if self.config.model.open_vision_clip_projector and name.startswith('clip_projector'):
                    logger.info(f"Unfreeze {name}")
                else:
                    logger.info(f"Freeze {name}")
                    p.requires_grad = False
        if self.config.model.freeze_text:
            for name, p in self.text_encoder.named_parameters():
                if self.config.model.open_text_projection and name.startswith('projection_layer'):
                    logger.info(f"Unfreeze {name}")
                else:
                    logger.info(f"Freeze {name}")
                    p.requires_grad = False

        img_size = self.config.model.vision_encoder.img_size
        self.transform = transforms.Compose(
            [
                transforms.Resize(
                    (img_size, img_size),
                    interpolation=InterpolationMode.BICUBIC,
                ),
                transforms.Lambda(lambda x: x.float().div(255.0)),
                transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
            ]
        )
        
        # load pretrained models
        self.load_checkpoint(
            config.model.vision_ckpt_path, config.model.text_ckpt_path, 
            config.model.get("extra_ckpt_path", None)
        )
        
        # criterions
        self.clip_loss = VTC_VTM_Loss(False)

    def no_weight_decay(self):
        ret = {"temp"}
        ret.update(
            {"vision_encoder." + k for k in self.vision_encoder.no_weight_decay()}
        )
        # no weight decay for LLM if training
        ret.update(
            {"text_encoder." + k for k, _ in self.text_encoder.named_parameters()}
        )

        return ret
    
    @torch.no_grad()
    def clip_contrastive_temperature(self):
        """Seems only used during pre-training"""
        self.temp.clamp_(min=self.temp_min)

    def forward(self, image, text, idx):
        """forward and calculate loss.

        Args:
            image (torch.Tensor): The input images. Shape: [B,T,C,H,W].
            text (dict): TODO
            idx (torch.Tensor): TODO

        Returns: TODO

        """
        self.clip_contrastive_temperature()
        vision_embeds = self.encode_vision(image)
        text_embeds = self.encode_text(text)

        # VTC loss
        loss_vtc = self.clip_loss.vtc_loss(
            vision_embeds, text_embeds, idx, self.temp, all_gather=True
        )

        return dict(
            loss_vtc=loss_vtc,
        )

    def encode_vision(self, image, test=False):
        """encode image / videos as features.

        Args:
            image (torch.Tensor): The input images.
            test (bool): Whether testing.

        Returns: tuple.
            - vision_embeds (torch.Tensor): The features of all patches. Shape: [B,C].

        """
        T = image.shape[1]
        use_image = True if T == 1 else False
        image = image.permute(0, 2, 1, 3, 4) # [B,T,C,H,W] -> [B,C,T,H,W]

        vision_embeds = self.vision_encoder(image, use_image=use_image)
        vision_embeds = self.vision_align(vision_embeds)
        return vision_embeds

    def encode_text(self, text):
        """encode text.
        Args:
            text (dict): The output of huggingface's `PreTrainedTokenizer`. contains keys:
                - input_ids (torch.Tensor): Token ids to be fed to a model. Shape: [B,L].
                - attention_mask (torch.Tensor): The mask indicate padded tokens. Shape: [B,L]. 0 is padded token.
                - other keys refer to "https://huggingface.co/docs/transformers/v4.21.2/en/main_classes/tokenizer#transformers.PreTrainedTokenizer.__call__".
        Returns: tuple.
            - text_embeds (torch.Tensor): The features of all tokens. Shape: [B,C].

        """
        text_embeds = self.text_encoder(text)
        return text_embeds

    def build_vision_encoder(self):
        """build vision encoder
        Returns: (vision_encoder, vision_layernorm). Each is a `nn.Module`.

        """
        vision_encoder = InternVideo2(
            in_chans=self.config.model.vision_encoder.in_chans,
            patch_size=self.config.model.vision_encoder.patch_size,
            img_size=self.config.model.vision_encoder.img_size,
            qkv_bias=self.config.model.vision_encoder.qkv_bias,
            drop_path_rate=self.config.model.vision_encoder.drop_path_rate,
            head_drop_path_rate=self.config.model.vision_encoder.head_drop_path_rate,
            embed_dim=self.config.model.vision_encoder.embed_dim,
            num_heads=self.config.model.vision_encoder.num_heads,
            mlp_ratio=self.config.model.vision_encoder.mlp_ratio,
            init_values=self.config.model.vision_encoder.init_values,
            qk_normalization=self.config.model.vision_encoder.qk_normalization,
            depth=self.config.model.vision_encoder.depth,
            use_flash_attn=self.config.model.vision_encoder.use_flash_attn,
            use_fused_rmsnorm=self.config.model.vision_encoder.use_fused_rmsnorm,
            use_fused_mlp=self.config.model.vision_encoder.use_fused_mlp,
            fused_mlp_heuristic=self.config.model.vision_encoder.fused_mlp_heuristic,
            attn_pool_num_heads=self.config.model.vision_encoder.attn_pool_num_heads,
            clip_embed_dim=self.config.model.vision_encoder.clip_embed_dim,
            layerscale_no_force_fp32=self.config.model.vision_encoder.layerscale_no_force_fp32,
            num_frames=self.config.model.vision_encoder.num_frames,
            tubelet_size=self.config.model.vision_encoder.tubelet_size,
            sep_pos_embed=self.config.model.vision_encoder.sep_pos_embed,
            use_checkpoint=self.config.model.vision_encoder.use_checkpoint,
            checkpoint_num=self.config.model.vision_encoder.checkpoint_num,
        )
        return vision_encoder

    def build_text_encoder(self, cfg, projection_dim):
        """build text_encoder and possiblly video-to-text multimodal fusion encoder.
        Returns: nn.Module. The text encoder

        """
        text_encoder = TextTransformer(cfg, projection_dim)

        return text_encoder

    def load_checkpoint(self, vision_ckpt_path=None, text_ckpt_path=None, extra_ckpt_path=None):
        assert vision_ckpt_path is not None, "No vision_encoder checkpoint"
        assert text_ckpt_path is not None, "No text_encoder checkpoint"

        new_ckpt = {}

        # load vision_encoder
        logger.info(f"Load vision_encoder checkpoint from {vision_ckpt_path}")
        vision_ckpt = torch.load(vision_ckpt_path, map_location='cpu')
        if 'module' in vision_ckpt.keys():
            vision_ckpt = vision_ckpt['module']
        elif 'model' in vision_ckpt.keys():
            vision_ckpt = vision_ckpt['model']
        if self.config.model.get('load_vision_ckpt_from_internvideo2_stage2', False):
            from .backbones.internvideo2.pos_embed import interpolate_pos_embed
            orig_t_size = self.config.model.get('vision_ckpt_t_size', 4)
            interpolate_pos_embed(vision_ckpt, self.vision_encoder, orig_t_size=orig_t_size) # 4 for InternVideo2 stage2
            for k, v in vision_ckpt.items():
                if k.startswith('vision_encoder.'):
                    if 'clip_decoder' in k or 'final_clip_decoder' in k:
                        continue
                    elif 'clip_pos_embed' in k or 'clip_img_pos_embed' in k or 'img_pos_embed' in k :
                        continue
                    else:
                        new_ckpt[k] = v
                else:
                    continue
        else:
            for k, v in vision_ckpt.items():
                if k.startswith('clip_decoder.') or k.startswith('mae_decoder.') or k.startswith('final_clip_decoder.'):
                    continue
                elif k in ['clip_pos_embed', 'mae_pos_embed']:
                    continue
                else:
                    new_k = 'vision_encoder.' + k
                    new_ckpt[new_k] = v

        # load text_encoder
        logger.info(f"Load text_encoder checkpoint from {text_ckpt_path}")
        test_ckpt = torch.load(text_ckpt_path, map_location='cpu')
        if 'module' in test_ckpt.keys():
            test_ckpt = test_ckpt['module']
        for k, v in test_ckpt.items():
            if k.startswith('text_encoder.'):
                new_ckpt[k] = v

        # load extra checkpoint
        # often when post-pretrain after previous pretraining, thus the keys are same
        if extra_ckpt_path is not None:
            logger.info(f"Load extra checkpoint from {extra_ckpt_path}")
            extra_ckpt = torch.load(extra_ckpt_path, map_location='cpu')
            if 'module' in extra_ckpt.keys():
                extra_ckpt = extra_ckpt['module']
            for k, v in extra_ckpt.items():
                new_ckpt[k] = v
        
        msg = self.load_state_dict(new_ckpt, strict=False)
        logger.info(msg)
