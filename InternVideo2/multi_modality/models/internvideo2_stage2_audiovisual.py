import logging

import torch
from torch import nn

from .backbones.beats.BEATs import BEATs, BEATsConfig
from .backbones.internvideo2 import pretrain_internvideo2_1b_patch14_224, pretrain_internvideo2_6b_patch14_224, internvl_clip_6b
from .backbones.bert.builder import build_bert
from .criterions import MLMLoss, VTC_VTM_Loss, new_UTA_Loss
from .mask import (
    TubeMaskingGenerator, 
    RandomMaskingGenerator
)

logger = logging.getLogger(__name__)


class InternVideo2_Stage2_audiovisual(nn.Module):
    """docstring for InternVideo2_Stage2_audiovisual"""

    def __init__(self, config, tokenizer, is_pretrain=True):
        super(InternVideo2_Stage2_audiovisual, self).__init__()

        self.config = config
        self.tokenizer = tokenizer

        self.is_pretrain = is_pretrain
        self.vision_width = config.model.vision_encoder.d_model
        self.vision_proj_width = config.model.vision_encoder.clip_embed_dim
        

        self.text_width = config.model.text_encoder.d_model
        self.audio_width = config.model.audio_encoder.d_model

        self.contra_dim = config.model.contra_dim
        self.av_concat_dim = config.model.av_concat_dim

        self.loss_weight = config.criterion.loss_weight
        self.loss_caption = config.criterion.loss_caption

        # create modules.
        self.vision_encoder = self.build_vision_encoder()
        if config.model.get("freeze_vision", False):
            self.freeze_vision()
        

        self.av_concat_vision_proj = nn.Sequential(nn.Linear(self.vision_width, self.av_concat_dim), nn.LayerNorm(self.av_concat_dim)) # NOTE for avtm & vtm

        self.text_encoder = self.build_text_encoder()
        if config.model.get("freeze_text", False):
            self.freeze_text()

        if self.use_audio():
            self.audio_encoder = self.build_audio_encoder()
            self.audio_proj = nn.Linear(self.audio_width, self.contra_dim) 
            self.av_concat_audio_proj = nn.Sequential(nn.Linear(self.audio_width, self.av_concat_dim), nn.LayerNorm(self.av_concat_dim))

            if self.loss_weight.avtc != 0 or self.loss_weight.avtm != 0:
                self.av_fusion = nn.Linear(2*self.contra_dim, self.contra_dim)

            if self.loss_weight.atm != 0:
                self.atm_head = nn.Linear(self.text_width, 2)

            if self.loss_weight.avtm != 0:
                self.avtm_head = nn.Linear(self.text_width, 2)
                
            # Freeze audio weights
            if config.model.get("freeze_audio", False):
                self.freeze_audio()

        self.vision_proj = nn.Linear(self.vision_proj_width, self.contra_dim)
        self.text_proj = nn.Linear(self.text_width, self.contra_dim)

        self.temp = nn.parameter.Parameter(torch.ones([]) * config.model.temp)
        self.itm_head = nn.Linear(self.text_width, 2)

        # criterions
        self.loss_weight = config.criterion.loss_weight
        self.criterion_uta = new_UTA_Loss(
            config.criterion.distill_final_features,
            config.criterion.clip_loss_ratio, 
        )
        self.criterion_vtc_vtm = VTC_VTM_Loss(config.criterion.vtm_hard_neg)
        self.criterion_mlm = MLMLoss(config.criterion.mlm_masking_prob, tokenizer)
        self.uta_image_only = config.criterion.get('uta_image_only', False)
        logger.info(f"uta_image_only={self.uta_image_only}")

        self.caption_set = set()
        for k in self.loss_caption.keys():
            if self.loss_weight[k] != 0:
                self.caption_set.add(self.loss_caption[k])
        
        logger.info(f"caption_set: {self.caption_set}")

        logger.info(f"unfreeze_keys: {config.model.get('unfreeze_keys', [])}")
        for k, p in self.named_parameters():
            for uk in config.model.get("unfreeze_keys", []):
                if uk in k: # vision_encoder.clip_projector
                    p.requires_grad = True
                    logger.info(f"unfreeze_key: {k}")

        self.num_test_segments = config.get("num_test_segments", 1)
        logger.info(f"num_test_segments={self.num_test_segments}")


    def use_audio(self):
        return self.loss_weight.atc != 0 or self.loss_weight.avc != 0 or self.loss_weight.avtc != 0 or self.loss_weight.atm != 0 or self.loss_weight.avtm != 0 or self.loss_weight.amlm != 0 or self.loss_weight.avmlm != 0 # or self.loss_weight.avstc != 0 or self.loss_weight.avstm != 0 or self.loss_weight.avsmlm != 0

    def freeze_vision(self):
        """freeze vision encoder"""
        for p in self.vision_encoder.parameters():
            p.requires_grad = False

    def freeze_text(self):
        """freeze text encoder"""
        for p in self.text_encoder.parameters():
            p.requires_grad = False

    def freeze_audio(self):
        """freeze audio encoder"""
        for p in self.audio_encoder.parameters():
            p.requires_grad = False




    def no_weight_decay(self):
        ret = {"temp"}
        ret.update(
            {"vision_encoder." + k for k in self.vision_encoder.no_weight_decay()}
        )
        # ret.update(
        #     {"text_encoder." + k for k in self.text_encoder.no_weight_decay()}
        # )

        return ret

    @property
    def dtype(self):
        return self.vision_encoder.patch_embed.proj.weight.dtype
    
    def forward_multi_captions(self, captions):
        text_dict = {}
        text_embeds_dict = {}
        text_proj_dict = {}

        for k in self.caption_set:
            text = captions[k]
            text_embeds, pooled_text_embeds = self.encode_text(text)
            text_proj = self.text_proj(pooled_text_embeds)
            text_dict[k] = text 
            text_embeds_dict[k] = text_embeds
            text_proj_dict[k] = text_proj
        
        return text_dict, text_embeds_dict, text_proj_dict

    def forward_audio(self, audio, text, idx):
        """forward and calculate loss."""
        assert self.use_audio(), self.loss_weight

        self.clip_contrastive_temperature()
        
        audio_embeds, pooled_audio_embeds = self.encode_audio(audio)
        text_embeds, pooled_text_embeds = self.encode_text(text)

        # obtain audio and text representations.
        audio_proj = self.audio_proj(pooled_audio_embeds)
        text_proj = self.text_proj(pooled_text_embeds)

        # calculate loss
        ## ATC loss
        if self.loss_weight.atc != 0:
            loss_atc = self.criterion_vtc_vtm.vtc_loss(
                audio_proj, text_proj, idx, self.temp, all_gather=True)
        else:
            loss_atc = torch.tensor(0)

        ## ATM loss
        if self.loss_weight.atm != 0:
            loss_atm = self.criterion_vtc_vtm.vtm_loss(
                self.get_text_encoder(),
                self.atm_head,
                self.temp,
                audio_embeds,
                text_embeds,
                audio_proj,
                text_proj,
                text.attention_mask,
                idx,
            )
        else:
            loss_atm = torch.tensor(0)

        ## AMLM loss
        if self.is_pretrain and self.loss_weight.mlm != 0:
            loss_amlm = self.criterion_mlm.mlm_loss(
                self.text_encoder, text, audio_embeds, None
            )
        else:
            loss_amlm = torch.tensor(0)

        return dict(
            loss_atc=loss_atc * self.loss_weight.atc,
            loss_atm=loss_atm * self.loss_weight.atm,
            loss_mlm=loss_amlm * self.loss_weight.amlm,
        )

    def forward_image_video(self, image, text, idx):
        """forward and calculate loss.

        Args:
            image (torch.Tensor): The input images. Shape: [B,T,C,H,W].
            text (dict): TODO
            idx (torch.Tensor): TODO

        Returns: TODO

        """
        
        self.clip_contrastive_temperature()
        T = image.shape[1]
        use_image = True if T == 1 else False

        vision_embeds, pooled_vision_embeds, student_output, student_output_final, targets_clip_middle_vis, targets_clip_final_vis = self.encode_vision(image)

        text_embeds, pooled_text_embeds = self.encode_text(text)

        # obtain vision and text representations.
        vision_proj = self.vision_proj(pooled_vision_embeds)
        text_proj = self.text_proj(pooled_text_embeds)

        # calculate loss
        ## UTA loss
        if self.loss_weight.uta != 0:
            if self.uta_image_only and not use_image:
                loss_uta = torch.tensor(0)
            else:
                loss_uta = self.criterion_uta.uta_loss(student_output, student_output_final, targets_clip_middle_vis, targets_clip_final_vis)
        else:
            loss_uta = torch.tensor(0)

        ## VTC loss
        if self.loss_weight.vtc != 0:
            loss_vtc = self.criterion_vtc_vtm.vtc_loss(
                vision_proj, text_proj, idx, self.temp, all_gather=True
            )
        else:
            loss_vtc = torch.tensor(0)

        ## VTM loss
        if self.loss_weight.vtm != 0:
            loss_vtm = self.criterion_vtc_vtm.vtm_loss(
                self.get_text_encoder(),
                self.itm_head,
                self.temp,
                vision_embeds,
                text_embeds,
                vision_proj,
                text_proj,
                text.attention_mask,
                idx,
            )
        else:
            loss_vtm = torch.tensor(0)

        ## MLM loss
        if self.is_pretrain and self.loss_weight.mlm != 0:
            loss_mlm = self.criterion_mlm.mlm_loss(
                self.text_encoder, text, vision_embeds, None
            )
        else:
            loss_mlm = torch.tensor(0)

        return dict(
            loss_uta=loss_uta * self.loss_weight.uta,
            loss_vtc=loss_vtc * self.loss_weight.vtc,
            loss_vtm=loss_vtm * self.loss_weight.vtm,
            loss_mlm=loss_mlm * self.loss_weight.mlm,
        )

    def forward_audio_video(self, audio, video, text, idx):
        """forward and calculate loss."""
        self.clip_contrastive_temperature()


        text_embeds, pooled_text_embeds = self.encode_text(text)
        text_proj = self.text_proj(pooled_text_embeds)

        # obtain vision and audio representations.
        vision_embeds, pooled_vision_embeds, student_output, student_output_final, targets_clip_middle_vis, targets_clip_final_vis = self.encode_vision(video)
        vision_proj = self.vision_proj(pooled_vision_embeds).squeeze(dim=1)
        
        if self.use_audio():
            audio_embeds, pooled_audio_embeds = self.encode_audio(audio)
            audio_proj = self.audio_proj(pooled_audio_embeds)

        # calculate loss
        ## UTA loss
        if self.loss_weight.uta != 0:
            if self.uta_image_only:
                loss_uta = torch.tensor(0)
            else:
                loss_uta = self.criterion_uta.uta_loss(student_output, student_output_final, targets_clip_middle_vis, targets_clip_final_vis)
        else:
            loss_uta = torch.tensor(0)

        ## VTC loss
        if self.loss_weight.vtc != 0:
            loss_vtc = self.criterion_vtc_vtm.vtc_loss(
                vision_proj, text_proj, idx, self.temp, all_gather=True
            )
        else:
            loss_vtc = torch.tensor(0)

        ## ATC loss
        if self.loss_weight.atc != 0:
            loss_atc = self.criterion_vtc_vtm.vtc_loss(
                audio_proj, text_proj, idx, self.temp, all_gather=True)
        else:
            loss_atc = torch.tensor(0)

        ## AVC loss
        if self.loss_weight.avc != 0:
            loss_avc = self.criterion_vtc_vtm.vtc_loss(
                audio_proj, vision_proj, idx, self.temp, all_gather=True)
        else:
            loss_avc = torch.tensor(0)


        ## AVTC loss
        if self.loss_weight.avtc != 0:
            loss_avtc = self.criterion_vtc_vtm.vtc_loss(
                self.av_fusion(torch.concat([audio_proj, vision_proj], dim=-1)), text_proj, idx, self.temp, all_gather=True
            )
        else:
            loss_avtc = torch.tensor(0)


        ## VTM loss
        if self.loss_weight.vtm != 0:
            loss_vtm = self.criterion_vtc_vtm.vtm_loss(
                self.get_text_encoder(),
                self.itm_head,
                self.temp,
                vision_embeds,
                text_embeds,
                vision_proj,
                text_proj,
                text.attention_mask,
                idx,
            )
        else:
            loss_vtm = torch.tensor(0)

        ## ATM loss
        if self.loss_weight.atm != 0:
            loss_atm = self.criterion_vtc_vtm.vtm_loss(
                self.get_text_encoder(),
                self.atm_head,
                self.temp,
                audio_embeds,
                text_embeds,
                audio_proj,
                text_proj,
                text.attention_mask,
                idx,
            )
        else:
            loss_atm = torch.tensor(0)

        ## AVTM loss
        if self.loss_weight.avtm != 0:
            loss_avtm = self.criterion_vtc_vtm.vtm_loss(
                self.get_text_encoder(),
                self.avtm_head,
                self.temp,
                torch.concat([audio_embeds, vision_embeds], dim=-2),
                text_embeds,
                self.av_fusion(torch.concat([audio_proj, vision_proj], dim=-1)),
                text_proj,
                text.attention_mask,
                idx,
            )
        else:
            loss_avtm = torch.tensor(0)


        ## MLM loss
        if self.is_pretrain and self.loss_weight.mlm != 0:
            loss_mlm = self.criterion_mlm.mlm_loss(
                self.text_encoder, text, vision_embeds, None
            )
        else:
            loss_mlm = torch.tensor(0)

        ## AMLM loss
        if self.is_pretrain and self.loss_weight.amlm != 0:
            loss_amlm = self.criterion_mlm.mlm_loss(
                self.text_encoder, text, audio_embeds, None
            )
        else:
            loss_amlm = torch.tensor(0)

        ## AVMLM loss
        if self.is_pretrain and self.loss_weight.avmlm != 0:
            loss_avmlm = self.criterion_mlm.mlm_loss(
                self.text_encoder, text, torch.concat([audio_embeds, vision_embeds], dim=-2), None
            )
        else:
            loss_avmlm = torch.tensor(0)



        return dict(
            loss_uta=loss_uta * self.loss_weight.uta,
            loss_vtc=loss_vtc * self.loss_weight.vtc,
            loss_vtm=loss_vtm * self.loss_weight.vtm,
            loss_mlm=loss_mlm * self.loss_weight.mlm,
            # audio-related
            loss_avc=loss_avc * self.loss_weight.avc,
            loss_atc=loss_atc * self.loss_weight.atc,
            loss_avtc=loss_avtc * self.loss_weight.avtc,
            loss_atm=loss_atm * self.loss_weight.atm,
            loss_avtm=loss_avtm * self.loss_weight.avtm,
            loss_amlm=loss_amlm * self.loss_weight.amlm,
            loss_avmlm=loss_avmlm * self.loss_weight.avmlm
            )

    def forward_audio_video_with_multi_captions(self, audio, video, text_dict, text_embeds_dict, text_proj_dict, idx):
        """forward and calculate loss."""
        self.clip_contrastive_temperature()

        # obtain vision and audio representations.
        vision_embeds, pooled_vision_embeds, student_output, student_output_final, targets_clip_middle_vis, targets_clip_final_vis = self.encode_vision(video)
        vision_proj = self.vision_proj(pooled_vision_embeds).squeeze(dim=1)
        
        if self.use_audio():
            audio_embeds, pooled_audio_embeds = self.encode_audio(audio)
            audio_proj = self.audio_proj(pooled_audio_embeds)

        # calculate loss
        ## UTA loss
        if self.loss_weight.uta != 0:
            if self.uta_image_only:
                loss_uta = torch.tensor(0)
            else:
                loss_uta = self.criterion_uta.uta_loss(student_output, student_output_final, targets_clip_middle_vis, targets_clip_final_vis)
        else:
            loss_uta = torch.tensor(0)

        ## VTC loss
        if self.loss_weight.vtc != 0:
            loss_vtc = self.criterion_vtc_vtm.vtc_loss(
                vision_proj, text_proj_dict[self.loss_caption['vtc']], idx, self.temp, all_gather=True
            )
        else:
            loss_vtc = torch.tensor(0)

        ## ATC loss
        if self.loss_weight.atc != 0:
            loss_atc = self.criterion_vtc_vtm.vtc_loss(
                audio_proj, text_proj_dict[self.loss_caption['atc']], idx, self.temp, all_gather=True)
        else:
            loss_atc = torch.tensor(0)

        ## AVC loss
        if self.loss_weight.avc != 0:
            loss_avc = self.criterion_vtc_vtm.vtc_loss(
                audio_proj, vision_proj, idx, self.temp, all_gather=True)
        else:
            loss_avc = torch.tensor(0)


        ## AVTC loss
        if self.loss_weight.avtc != 0:
            loss_avtc = self.criterion_vtc_vtm.vtc_loss(
                self.av_fusion(torch.concat([audio_proj, vision_proj], dim=-1)), text_proj_dict[self.loss_caption['avtc']], idx, self.temp, all_gather=True
            )
        else:
            loss_avtc = torch.tensor(0)


        ## VTM loss
        if self.loss_weight.vtm != 0:
            loss_vtm = self.criterion_vtc_vtm.vtm_loss(
                self.get_text_encoder(),
                self.itm_head,
                self.temp,
                vision_embeds,
                text_embeds_dict[self.loss_caption['vtm']],
                vision_proj,
                text_proj_dict[self.loss_caption['vtm']],
                text_dict[self.loss_caption['vtm']].attention_mask,
                idx,
            )
        else:
            loss_vtm = torch.tensor(0)

        ## ATM loss
        if self.loss_weight.atm != 0:
            loss_atm = self.criterion_vtc_vtm.vtm_loss(
                self.get_text_encoder(),
                self.atm_head,
                self.temp,
                audio_embeds,
                text_embeds_dict[self.loss_caption['atm']],
                audio_proj,
                text_proj_dict[self.loss_caption['atm']],
                text_dict[self.loss_caption['atm']].attention_mask,
                idx,
            )
        else:
            loss_atm = torch.tensor(0)

        ## AVTM loss
        if self.loss_weight.avtm != 0:
            loss_avtm = self.criterion_vtc_vtm.vtm_loss(
                self.get_text_encoder(),
                self.avtm_head,
                self.temp,
                torch.concat([audio_embeds, vision_embeds], dim=-2),
                text_embeds_dict[self.loss_caption['avtm']],
                self.av_fusion(torch.concat([audio_proj, vision_proj], dim=-1)),
                text_proj_dict[self.loss_caption['avtm']],
                text_dict[self.loss_caption['avtm']].attention_mask,
                idx,
            )
        else:
            loss_avtm = torch.tensor(0)


        ## MLM loss
        if self.is_pretrain and self.loss_weight.mlm != 0:
            loss_mlm = self.criterion_mlm.mlm_loss(
                self.text_encoder, text_dict[self.loss_caption['mlm']], vision_embeds, None
            )
        else:
            loss_mlm = torch.tensor(0)

        ## AMLM loss
        if self.is_pretrain and self.loss_weight.amlm != 0:
            loss_amlm = self.criterion_mlm.mlm_loss(
                self.text_encoder, text_dict[self.loss_caption['amlm']], audio_embeds, None
            )
        else:
            loss_amlm = torch.tensor(0)

        ## AVMLM loss
        if self.is_pretrain and self.loss_weight.avmlm != 0:
            loss_avmlm = self.criterion_mlm.mlm_loss(
                self.text_encoder, text_dict[self.loss_caption['avmlm']], torch.concat([audio_embeds, vision_embeds], dim=-2), None
            )
        else:
            loss_avmlm = torch.tensor(0)



        return dict(
            loss_uta=loss_uta * self.loss_weight.uta,
            loss_vtc=loss_vtc * self.loss_weight.vtc,
            loss_vtm=loss_vtm * self.loss_weight.vtm,
            loss_mlm=loss_mlm * self.loss_weight.mlm,
            # audio-related
            loss_avc=loss_avc * self.loss_weight.avc,
            loss_atc=loss_atc * self.loss_weight.atc,
            loss_avtc=loss_avtc * self.loss_weight.avtc,
            loss_atm=loss_atm * self.loss_weight.atm,
            loss_avtm=loss_avtm * self.loss_weight.avtm,
            loss_amlm=loss_amlm * self.loss_weight.amlm,
            loss_avmlm=loss_avmlm * self.loss_weight.avmlm
            )

    def forward(self, media, text, idx, media_type='image'):
        """forward and calculate loss."""

        if media_type == 'audio_video':
            audio = media[0]
            video = media[1]
            if type(text) is dict:
                text_dict, text_embeds_dict, text_proj_dict = self.forward_multi_captions(text) # NOTE 应该只有audio video需要多个caption
                return self.forward_audio_video_with_multi_captions(audio, video, text_dict, text_embeds_dict, text_proj_dict, idx)
            else:
                return self.forward_audio_video(audio, video, text, idx)
            
        elif media_type == 'audio':
            return self.forward_audio(media, text, idx)
        elif media_type in ['image', 'video']:
            return self.forward_image_video(media, text, idx)
        else:
            raise NotImplementedError(f"Not support {media_type} now!!!")

    def encode_teacher(self, image):
        """encode image / videos as features.

        Args:
            image (torch.Tensor): The input images.

        Returns: tuple.
            - mask (torch.Tensor): Mask. Shape: [B,N1].
            - d_mask (torch.Tensor): Double Mask. Shape: [B,N2].
            - clip_output (torch.Tensor): The features of clip. Shape: [K,B,N,C].

        """
        B, C, T, H, W = image.shape
        mask_type = self.image_mask_type if T == 1 else self.video_mask_type
        window_size = self.image_window_size if T == 1 else self.video_window_size
        mask_ratio = self.image_mask_ratio if T == 1 else self.video_mask_ratio

        if (self.uta_image_only and T != 1) or self.config.model.vision_encoder.get('only_mask', False):
            if mask_type == 'tube':
                mask = TubeMaskingGenerator(window_size, mask_ratio, B)
            elif mask_type == 'random':
                mask = RandomMaskingGenerator(window_size, mask_ratio, B)
            elif mask_type == 'none':
                return None, None, None
            else:
                raise NotImplementedError
            
            mask = mask.view(B, -1).to(torch.bool)
            mask = torch.cat((torch.zeros(B, 1).to(mask.device), mask), dim=1)
            mask = mask.to(torch.bool)

            return mask, None, None
        
        if self.clip_teacher is None or self.loss_weight.uta == 0:
            return None, None, None

        if H != self.clip_img_size:
            image = torch.nn.functional.interpolate(
                image.reshape(B, C*T, H, W), 
                size=(self.clip_img_size, self.clip_img_size), 
                mode='bicubic', align_corners=False
            )
            image = image.view(B, C, T, self.clip_img_size, self.clip_img_size)

        with torch.no_grad():
            if mask_type == 'tube':
                mask = TubeMaskingGenerator(window_size, mask_ratio, B)
                norm_clip_middle, norm_clip_final, attn = self.clip_teacher(image)
            elif mask_type == 'random':
                mask = RandomMaskingGenerator(window_size, mask_ratio, B)
                norm_clip_middle, norm_clip_final, attn = self.clip_teacher(image)
            elif mask_type in 'attention':
                norm_clip_middle, norm_clip_final, attn = self.clip_teacher(image)
                BT, N = attn.shape
                N_vis = N - int(N * mask_ratio)
                importance = torch.multinomial(attn, N)
                mask = torch.ones((BT, N))
                pos1 = torch.arange(BT).view(-1, 1).repeat(1, N_vis)
                pos2 = importance[:, :N_vis]
                mask[pos1, pos2] = 0
            else:
                raise NotImplementedError
            
            mask = mask.view(B, -1).to(torch.bool)
            mask = torch.cat((torch.zeros(B, 1), mask), dim=1)
            mask = mask.to(torch.bool)

            # mask clip output
            C_CLIP = norm_clip_middle.shape[-1]
            if len(norm_clip_middle.shape) == 4:
                K = norm_clip_middle.shape[0]
                clip_mask = mask.unsqueeze(0).repeat(K, 1, 1)
                targets_clip_middle_vis = norm_clip_middle[~clip_mask].reshape(K, B, -1, C_CLIP)
            else:
                clip_mask = mask
                targets_clip_middle_vis = norm_clip_middle[~clip_mask].reshape(B, -1, C_CLIP)
                
            targets_clip_final_vis = norm_clip_final # only one tokens

        return mask, targets_clip_middle_vis, targets_clip_final_vis

    def encode_vision(self, image, test=False):
        """encode image / videos as features.

        Args:
            image (torch.Tensor): The input images.
            test (bool): Whether testing.

        Returns: tuple.
            - vision_embeds (torch.Tensor): The output features. Shape: [B,N,C].
            - pooled_vision_embeds (torch.Tensor): The pooled output features. Shape: [B,1,C].
            - student_output (torch.Tensor): The features of alignment. Shape: [K,B,N,C].
            - clip_output (torch.Tensor): The features of clip. Shape: [K,B,N,C].

        """
        B, T, C, H, W = image.shape
        use_image = True if T == 1 else False
        if not use_image and test and self.num_test_segments != 1:
            assert T // self.num_test_segments == self.config.model.vision_encoder.num_frames 
            # print(f"origin image.shape: {image.shape}")
            image = image.view(B, T // self.num_test_segments, self.num_test_segments, C, H, W).permute(0, 2, 1, 3, 4, 5).reshape(B * self.num_test_segments, T // self.num_test_segments, C, H, W) # [B,T,C,H,W] -> [B*num_test_segments,T // num_test_segments,C,H,W]
            # print(f"new image.shape: {image.shape}")

        image = image.permute(0, 2, 1, 3, 4) # [B,T,C,H,W] -> [B,C,T,H,W]
        # whether save temporal dimension
        # keep_temporal=self.config.model.vision_encoder.keep_temporal
        if test:
            vision_embeds, pooled_vision_embeds, _, _ = self.vision_encoder(
                image, None, use_image)
            vision_embeds = self.av_concat_vision_proj(vision_embeds) # keep consistent with av conat
            # print(f"vision_embeds.shape is {vision_embeds.shape} {pooled_vision_embeds.shape}")
            # torch.Size([4, 1025, av_concat_dim]) torch.Size([4, 768])

            if not use_image and self.num_test_segments != 1:
                n, d = vision_embeds.shape[-2], vision_embeds.shape[-1]
                # print(f"origin vision_embeds.shape: {vision_embeds.shape}")
                vision_embeds = vision_embeds.reshape(B, self.num_test_segments, n, d).mean(1, keepdim=False)
                # print(f"new vision_embeds.shape: {vision_embeds.shape}")
                # print(f"origin pooled_vision_embeds.shape: {pooled_vision_embeds.shape}")
                d = pooled_vision_embeds.shape[-1]
                pooled_vision_embeds = pooled_vision_embeds.reshape(B, self.num_test_segments, d).mean(1, keepdim=False)
                # print(f"new pooled_vision_embeds.shape: {pooled_vision_embeds.shape}")

            return vision_embeds, pooled_vision_embeds
        else:
            mask, targets_clip_middle_vis, targets_clip_final_vis = self.encode_teacher(image) 
            # if mask is not None and (self.video_mask_type != 'tube' or self.image_mask_type != 'tube'):
            #     keep_temporal = False

            vision_embeds, pooled_vision_embeds, student_output, student_output_final = self.vision_encoder(
                    image, mask, use_image)
            vision_embeds = self.av_concat_vision_proj(vision_embeds) # keep consistent with av conat
            return vision_embeds, pooled_vision_embeds, student_output, student_output_final, targets_clip_middle_vis, targets_clip_final_vis

    def encode_audio(self, audio, test=False):
        audio_embeds = self.audio_encoder(audio) # 16, 248, 768 only use fank
        pooled_audio_embeds = audio_embeds.mean(dim=1) # NOTE pool计算放conat_proj前面
        audio_embeds = self.av_concat_audio_proj(audio_embeds) # keep consistent with av conat
        return audio_embeds, pooled_audio_embeds
    
    def encode_text(self, text):
        """encode text.
        Args:
            text (dict): The output of huggingface's `PreTrainedTokenizer`. contains keys:
                - input_ids (torch.Tensor): Token ids to be fed to a model. Shape: [B,L].
                - attention_mask (torch.Tensor): The mask indicate padded tokens. Shape: [B,L]. 0 is padded token.
                - other keys refer to "https://huggingface.co/docs/transformers/v4.21.2/en/main_classes/tokenizer#transformers.PreTrainedTokenizer.__call__".
        Returns: tuple.
            - text_embeds (torch.Tensor): The features of all tokens. Shape: [B,L,C].
            - pooled_text_embeds (torch.Tensor): The pooled features. Shape: [B,C].

        """
        text_output = self.get_text_encoder()(
            text.input_ids,
            attention_mask=text.attention_mask,
            return_dict=True,
            mode="text",
        )
        text_embeds = text_output.last_hidden_state
        pooled_text_embeds = text_embeds[:, 0]
        return text_embeds, pooled_text_embeds

    @torch.no_grad()
    def clip_contrastive_temperature(self, min_val=0.001, max_val=0.5):
        """Seems only used during pre-training"""
        self.temp.clamp_(min_val, max_val)

    def build_vision_encoder(self):
        """build vision encoder
        Returns: (vision_encoder, clip_teacher). Each is a `nn.Module`.

        """
        encoder_name = self.config.model.vision_encoder.name
        logger.info(f"Build vision_encoder: {encoder_name}")
        if encoder_name == 'pretrain_umt2_giant_patch14_224':
            vision_encoder = pretrain_umt2_giant_patch14_224(self.config.model)
        elif encoder_name == 'pretrain_umt2_giant_patch14_224_clean':
            vision_encoder = pretrain_umt2_giant_patch14_224_clean(self.config.model)
        elif encoder_name == 'pretrain_umt2_6b_patch14_224':
            vision_encoder = pretrain_umt2_6b_patch14_224(self.config.model)
        elif encoder_name == 'pretrain_umt2_6b_patch14_224_clean':
            vision_encoder = pretrain_umt2_6b_patch14_224_clean(self.config.model)
        else:
            raise ValueError(f"Not implemented: {encoder_name}")

        teacher_name = self.config.model.vision_encoder.clip_teacher
        self.clip_teacher = None
        if teacher_name != None and teacher_name != "none":
            assert teacher_name == 'internvl_clip_6b'
            self.clip_teacher = internvl_clip_6b(
                img_size=self.config.model.vision_encoder.clip_input_resolution,
                clip_norm_type=self.config.model.vision_encoder.clip_norm_type,
                return_attn=True,
                clip_return_layer=self.config.model.vision_encoder.clip_return_layer,
                clip_return_interval=self.config.model.vision_encoder.clip_teacher_return_interval
                )
            for p in self.clip_teacher.parameters():
                p.requires_grad = False

        # parameters for mask
        img_size = self.config.model.vision_encoder.img_size
        num_frames = self.config.model.vision_encoder.num_frames
        tublet_size = self.config.model.vision_encoder.tubelet_size
        patch_size = self.config.model.vision_encoder.patch_size
        self.clip_img_size = self.config.model.vision_encoder.clip_input_resolution
        self.video_mask_type = self.config.model.vision_encoder.video_mask_type
        self.video_window_size = (num_frames // tublet_size, img_size // patch_size, img_size // patch_size)
        self.video_mask_ratio = self.config.model.vision_encoder.video_mask_ratio
        self.video_double_mask_ratio = self.config.model.vision_encoder.video_double_mask_ratio
        self.image_mask_type = self.config.model.vision_encoder.image_mask_type
        self.image_window_size = (1, img_size // patch_size, img_size // patch_size)
        self.image_mask_ratio = self.config.model.vision_encoder.image_mask_ratio
        self.image_double_mask_ratio = self.config.model.vision_encoder.image_double_mask_ratio
        
        return vision_encoder

    def build_text_encoder(self):
        """build text_encoder and possiblly video-to-text multimodal fusion encoder.
        Returns: nn.Module. The text encoder

        """
        encoder_name = self.config.model.text_encoder.name
        logger.info(f"Build text_encoder {encoder_name}")

        if "bert" in encoder_name:
            text_encoder = build_bert(
                self.config.model,
                self.is_pretrain,
                self.config.gradient_checkpointing,
                encoder_width=self.av_concat_dim
            )
        else:
            raise ValueError(f"Not implemented: {encoder_name}")

        return text_encoder

    def get_text_encoder(self):
        """get text encoder, used for text and cross-modal encoding"""
        encoder = self.text_encoder
        return encoder.bert if hasattr(encoder, "bert") else encoder

    def build_audio_encoder(self):
        audio_model_path = self.config.model.audio_encoder.get('audio_model_path', None)
        assert audio_model_path is not None, "You must load pretrained model!"
        logger.info(f"Load audio model from {audio_model_path}")
        if self.config.model.audio_encoder.name == 'beats':
            checkpoint = torch.load(audio_model_path, map_location="cpu")
            audio_cfg = BEATsConfig(checkpoint['cfg'])
            BEATs_model = BEATs(audio_cfg)
            msg = BEATs_model.load_state_dict(checkpoint['model'])
            logger.info(msg)
            return BEATs_model
        elif self.config.model.audio_encoder.name == 'beats_no_weight_norm_debug':
            checkpoint = torch.load(audio_model_path, map_location="cpu")
            audio_cfg = BEATsConfig(checkpoint['cfg'])
            BEATs_model = BEATs(audio_cfg)
            return BEATs_model
        else:
            raise NotImplementedError(self.config.model.audio_encoder.name)