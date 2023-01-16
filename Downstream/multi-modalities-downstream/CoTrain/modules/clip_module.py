from copy import deepcopy
import torch
import torch.nn as nn
import pytorch_lightning as pl
from CoTrain.modules import heads, cotrain_utils
from CoTrain.modules import objectives as objectives
from CoTrain.modules import base_vision_transformer as vit
from CoTrain.modules.text_prompt import text_prompt
import os
import matplotlib.pyplot as plt
import math

import CoTrain.modules.InternVideo as clip_kc_new
from PIL import Image
import numpy as np
from .clip_param_keys import clip_param_keys, gradually_freeze_by_layer
from .clip_decoders import CaptionDecoder


def vis_save(imgs, texts):
    # img: [B, T, C, H, W]
    # texts: [str]
    os.makedirs("vis_test", exist_ok=True)
    imgs = imgs.permute(0, 1, 3, 4, 2).cpu().numpy()
    imgs = (imgs - imgs.min()) / (imgs.max() - imgs.min())
    for img, text in zip(imgs, texts):
        caption = "_".join(text.split())
        os.makedirs(os.path.join("vis_test", caption), exist_ok=True)
        for i, im in enumerate(img):
            img_path = os.path.join("vis_test", caption, f"{i}.png")
            plt.imsave(img_path, im)


# utils
@torch.no_grad()
def concat_all_gather(tensor):
    """
    Performs all_gather operation on the provided tensors.
    *** Warning ***: torch.distributed.all_gather has no gradient.
    """
    tensors_gather = [
        torch.ones_like(tensor) for _ in range(torch.distributed.get_world_size())
    ]
    torch.distributed.all_gather(tensors_gather, tensor, async_op=False)

    output = torch.cat(tensors_gather, dim=0)
    return output


@torch.no_grad()
def batch_shuffle_ddp(x):
    """
    Batch shuffle, for making use of BatchNorm.
    *** Only support DistributedDataParallel (DDP) model. ***
    """
    # gather from all gpus
    batch_size_this = x.shape[0]
    x_gather = concat_all_gather(x)
    batch_size_all = x_gather.shape[0]

    num_gpus = batch_size_all // batch_size_this

    # random shuffle index
    idx_shuffle = torch.randperm(batch_size_all).cuda()

    # broadcast to all gpus
    torch.distributed.broadcast(idx_shuffle, src=0)

    # index for restoring
    idx_unshuffle = torch.argsort(idx_shuffle)

    # shuffled index for this gpu
    gpu_idx = torch.distributed.get_rank()
    idx_this = idx_shuffle.view(num_gpus, -1)[gpu_idx]

    return x_gather[idx_this], idx_unshuffle


@torch.no_grad()
def batch_unshuffle_ddp(x, idx_unshuffle):
    """
    Undo batch shuffle.
    *** Only support DistributedDataParallel (DDP) model. ***
    """
    # gather from all gpus
    batch_size_this = x.shape[0]
    x_gather = concat_all_gather(x)
    batch_size_all = x_gather.shape[0]

    num_gpus = batch_size_all // batch_size_this

    # restored index for this gpu
    gpu_idx = torch.distributed.get_rank()
    idx_this = idx_unshuffle.view(num_gpus, -1)[gpu_idx]

    return x_gather[idx_this]


class CLIP(pl.LightningModule):
    def __init__(self, config):
        super().__init__()
        self.save_hyperparameters()
        self.clip_type = config["clip_type"]
        self.prompt_type = config["prompt_type"]
        self.mlm_prob = config["mlm_prob"]
        self.mim_prob = config["mim_prob"]
        self.qa_type = config["clip_qa_type"]
        self.mc_type = config["clip_mc_type"]
        self.mmt = config["clip_mmt"]
        self.alt_data = config["clip_alt_data"]

        if config["clip_type"] == "kc_new":
            self.clip, self.clip_preprocess = clip_kc_new.load(
                config["clip"],
                t_size=config["num_frames"],
                n_layers=4,
                mlp_dropout=[config["clip_evl_dropout"]] * 4,
                cls_dropout=config["clip_evl_dropout"],
                no_pretrain=config["clip_no_pretrain"],
                init_zero=config["clip_init_zero"],
                drop_path_rate=config["clip_dpr"],
                device=self.device,
                use_checkpoint=config["clip_use_checkpoint"],
                checkpoint_num=config["clip_checkpoint_num"],
                # mask_text=(
                #     self.hparams.config["loss_names"]["mlm"] > 0
                #     or (
                #         self.hparams.config["loss_names"]["openend_vqa"] > 0
                #         and self.qa_type in ["zs", "mlm", "vtc_mlm"]
                #     )
                # ),
            )
        else:
            raise NotImplementedError(
                "Clip type: {} not implemented".format(config["clip_type"])
            )

        cotrain_utils.set_metrics(self)
        self.current_tasks = list()

        vision_width = self.clip.visual.conv1.weight.shape[0]
        transformer_width = self.clip.transformer.width

        if self.hparams.config["loss_names"]["openend_vqa"] > 0:
            if self.qa_type == "vtc":
                hs = vision_width + transformer_width
            elif self.qa_type in ["cap"]:
                hs = transformer_width
            elif self.qa_type in ["vtc_cap", "vtc_mlm"]:
                # We cat the vision feature, text feature
                # and cross feature together
                hs = vision_width + transformer_width * 2
            elif self.qa_type in ["zs", "mlm"]:
                pass
            else:
                raise NotImplementedError("QA Type {} Not Implemented")
            if self.qa_type in ["vtc", "cap", "vtc_cap", "vtc_mlm"]:
                self.clip.text_projection = None
                self.clip.visual_proj = None
                vs = self.hparams.config["msrvttqa_label_size"]
                self.vqa_classifier = nn.Sequential(
                    nn.Dropout(config["clip_cls_dropout"]),
                    nn.Linear(hs, hs * 2),
                    nn.LayerNorm(hs * 2),
                    nn.GELU(),
                    nn.Dropout(config["clip_cls_dropout"]),
                    # nn.Linear(hs * 2, hs * 2),
                    # nn.GELU(),
                    # nn.Dropout(config["clip_cls_dropout"]),
                    # nn.LayerNorm(hs * 2),
                    nn.Linear(hs * 2, vs),
                )
                self.vqa_classifier.apply(objectives.init_weights)

        if self.hparams.config["loss_names"]["multiple_choice"] > 0:
            if self.mc_type == "vtc":
                pass
            elif self.mc_type == "cap":
                hs = transformer_width
            elif self.mc_type == "vtc_cap":
                # We cat the vision feature, text feature
                # and cross feature together
                hs = vision_width + transformer_width * 2
            else:
                raise NotImplementedError("MC Type {} Not Implemented")

            if self.mc_type in ["cap", "vtc_cap"]:
                self.clip.text_projection = None
                self.clip.visual_proj = None
                self.rank_output = nn.Sequential(
                    nn.Dropout(config["clip_cls_dropout"]),
                    nn.Linear(hs, hs * 2),
                    nn.LayerNorm(hs * 2),
                    nn.GELU(),
                    nn.Dropout(config["clip_cls_dropout"]),
                    nn.Linear(hs * 2, 1),
                )
                self.rank_output.apply(objectives.init_weights)

        if (
            self.hparams.config["loss_names"]["cap"] > 0
            or (
                self.hparams.config["loss_names"]["openend_vqa"] > 0
                and self.qa_type in ["cap", "vtc_cap"]
            )
            or (
                self.hparams.config["loss_names"]["multiple_choice"] > 0
                and self.mc_type in ["cap", "vtc_cap"]
            )
        ):
            vs = self.clip.vocab_size
            if self.hparams.config["loss_names"][
                "openend_vqa"
            ] > 0 and self.qa_type in ["cap", "vtc_cap"]:
                vs = self.hparams.config["msrvttqa_label_size"]
            self.caption_decoder = CaptionDecoder(
                n_layers=config["clip_cap_decoder_n_layers"],
                transformer_width=transformer_width,
                vision_width=vision_width,
                transformer_heads=transformer_width // 64,
                vocab_size=vs,
                use_checkpoint=config["clip_use_checkpoint"],
                checkpoint_num=config["clip_checkpoint_num"],
            )
            if (
                self.hparams.config["loss_names"]["openend_vqa"] > 0
                and self.qa_type in ["cap", "vtc_cap"]
            ) or (
                self.hparams.config["loss_names"]["multiple_choice"] > 0
                and self.mc_type in ["cap", "vtc_cap"]
            ):
                self.caption_decoder.predictor = nn.Identity()
            self.caption_decoder.apply(objectives.init_weights)

        # For zs_classify
        self.text_ret = None

        if self.hparams.config["load_path"] != "":
            # Support multiple load_path
            if isinstance(self.hparams.config["load_path"], str):
                self.hparams.config["load_path"] = [self.hparams.config["load_path"]]

            for i, load_path in enumerate(self.hparams.config["load_path"]):
                ckpt = torch.load(
                    cotrain_utils.read_load_path(load_path),
                    map_location="cpu",
                )
                if i == 0:
                    state_dict = ckpt["state_dict"]
                    continue
                for k in state_dict.keys():
                    state_dict[k] += ckpt["state_dict"][k]

            for k in state_dict.keys():
                state_dict[k] /= len(self.hparams.config["load_path"])

            modified_keys = []
            if config["clip_wiseft_coef"] > 0:
                c = config["clip_wiseft_coef"]
                assert 0 < c < 1.0
                # We assume using clip weight by default
                clip_sd = {k: v.cpu() for k, v in self.state_dict().items()}
                new_sd = deepcopy(state_dict)
                # Directly modify state_dict to load
                for k in new_sd:
                    if k not in clip_sd:
                        continue
                    if any(x in k for x in clip_param_keys):
                        new_sd[k] = clip_sd[k] * c + state_dict[k] * (1.0 - c)
                        modified_keys.append(k)
                state_dict = new_sd

            # Remove mismatch parameters for 336
            sd = {k: v.cpu() for k, v in self.state_dict().items()}
            for k in list(state_dict.keys()):
                if k not in sd:
                    continue
                if state_dict[k].shape != sd[k].shape:
                    print(
                        "!!!!!!!!!!!Size mismatch {} {} {}".format(
                            k, state_dict[k].shape, sd[k].shape
                        )
                    )
                    del state_dict[k]

            self.load_state_dict(state_dict, strict=False)

        if config["clip_freeze"] and config["clip_type"] == "evl":
            self.freeze_clip_evl()

        if config["clip_freeze"] and config["clip_type"] == "kc":
            self.freeze_clip()

        if config["clip_freeze_text"]:
            self.freeze_text()

        self.grad_unfreeze_int = config["clip_grad_unfreeze_int"]
        if self.grad_unfreeze_int > 0:
            self.freeze_clip()

        if self.mmt:
            # MoCo Setting
            K = 65536
            m = 0.999
            dim = self.clip.embed_dim
            self.K = K
            self.m = m

            self.clip_k = deepcopy(self.clip)
            for p in self.clip_k.parameters():
                p.requires_grad = False  # not update by gradient

            # create the queue
            self.register_buffer("queue_visual", torch.randn(dim, K))
            self.register_buffer("queue_text", torch.randn(dim, K))
            self.queue_visual = nn.functional.normalize(self.queue_visual, dim=0)
            self.queue_text = nn.functional.normalize(self.queue_text, dim=0)

            self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))

    def freeze_clip_evl(self):
        for n, p in self.named_parameters():
            if (
                "clip.visual" in n
                and "clip.visual.ln_post" not in n
                and "clip.visual.proj" not in n
            ):
                p.requires_grad = False
            elif "clip.transformer" in n:
                p.requires_grad = False
            elif "clip.token_embedding" in n:
                p.requires_grad = False
            elif "clip.positional_embedding" in n:
                p.requires_grad = False

    def freeze_clip(self):
        for n, p in self.named_parameters():
            # Unfreeze the projection layer
            if any(x in n for x in ["text_projection", "visual_proj", "visual.proj"]):
                continue
            if any(x in n for x in clip_param_keys):
                p.requires_grad = False

    def freeze_text(self):
        for n, p in self.named_parameters():
            if "clip.transformer" in n:
                p.requires_grad = False
            elif "clip.token_embedding" in n:
                p.requires_grad = False
            elif "clip.positional_embedding" in n:
                p.requires_grad = False
            elif "clip.ln_final" in n:
                p.requires_grad = False
            elif "clip.text_projection" in n:
                p.requires_grad = False
            elif "clip.eot_token_embedding" in n:
                p.requires_grad = False

    @torch.no_grad()
    def mask_text_ids(self, text_ids, special_tokens_mask):
        if "openend_vqa" in self.current_tasks:
            return self.mask_text_ids_qa_mlm(text_ids)
        # See https://github.com/huggingface/transformers/blob/a22db885b41b3a1b302fc206312ee4d99cdf4b7c/src/transformers/data/data_collator.py#L748
        # text_ids, special_tokens_mask: torch.Tensor of shape (N, L)
        labels = text_ids.clone().long()
        probability_matrix = torch.full(labels.shape, self.mlm_prob, device=self.device)

        # do not mask special_token, including sot_token, eot_token and empty
        probability_matrix.masked_fill_(special_tokens_mask, value=0.0)
        # probability_matrix[:, 0] = 0.0
        # probability_matrix[torch.arange(labels.shape[0]), text_ids.argmax(dim=-1)] = 0.0

        masked_indices = torch.bernoulli(probability_matrix).bool()
        # We only compute loss on masked tokens, note that padding id is 0
        labels[~masked_indices] = -100

        # ? Should we use other augmentation in Bert?
        return text_ids, labels, masked_indices

    @torch.no_grad()
    def mask_text_ids_qa_mlm(self, text_ids):
        # The text should be in the format "Question: {} Anwser:"
        # We add a mask at the end of the sentence
        eot_id = text_ids[torch.arange(text_ids.shape[0]), text_ids.argmax(dim=-1)]
        assert torch.numel(torch.unique(eot_id)) == 1
        masked_indices = torch.zeros(
            *text_ids.shape, dtype=torch.bool, device=text_ids.device
        )
        masked_indices[torch.arange(text_ids.shape[0]), text_ids.argmax(dim=-1)] = 1

        labels = text_ids.clone()
        text_ids[torch.arange(labels.shape[0]), labels.argmax(dim=-1)] = eot_id - 1
        text_ids[torch.arange(labels.shape[0]), labels.argmax(dim=-1) + 1] = eot_id

        return text_ids, None, masked_indices

    @torch.no_grad()
    def mask_visual(self, video, mode="video"):
        assert mode in ["video", "image"]
        N, C, T, H, W = video.shape
        patch_size = self.clip.visual.conv1.weight.shape[-1]
        N = N * T
        L = H * W // (patch_size * patch_size)

        # This is different from text as we are masking a fix number of tokens
        Lm = int(self.mim_prob * L)
        masked_indices = torch.zeros(N, L)
        indices = torch.argsort(torch.rand_like(masked_indices), dim=-1)[:, :Lm]
        batch_indices = (
            torch.arange(masked_indices.shape[0]).unsqueeze(-1).expand_as(indices)
        )
        masked_indices[batch_indices, indices] = 1

        masked_indices = masked_indices.bool()
        return masked_indices

    @torch.no_grad()
    def _momentum_update_key_encoder(self):
        """
        Momentum update of the key encoder
        """
        for param_q, param_k in zip(self.clip.parameters(), self.clip_k.parameters()):
            param_k.data = param_k.data * self.m + param_q.data * (1.0 - self.m)

    @torch.no_grad()
    def _dequeue_and_enqueue(self, text_keys, visual_keys):
        # gather keys before updating queue
        text_keys = concat_all_gather(text_keys)
        visual_keys = concat_all_gather(visual_keys)

        batch_size = text_keys.shape[0]

        ptr = int(self.queue_ptr)
        assert self.K % batch_size == 0  # for simplicity

        # replace the keys at ptr (dequeue and enqueue)
        self.queue_text[:, ptr : ptr + batch_size] = text_keys.T
        self.queue_visual[:, ptr : ptr + batch_size] = visual_keys.T
        ptr = (ptr + batch_size) % self.K  # move pointer

        self.queue_ptr[0] = ptr

    def infer(
        self,
        batch,
        mask_text=False,
        mask_video=False,
        input_video_only=False,
        input_text_only=False,
        caption=False,
        mode="video",
    ):
        imgkey = "video"

        # Check configs
        assert not input_video_only
        assert not input_text_only
        if mask_text:
            assert self.clip_type in ["ori", "evl", "kc", "kc_new"]
        if mask_video:
            assert self.clip_type in ["ori", "kc", "kc_new"]

        # Encode Text #########################################################
        if "clip_text_ids" in batch:
            # If the clip tokenization is prepared
            text_ids, special_tokens_mask = (
                batch["clip_text_ids"],
                batch["clip_special_tokens_mask"],
            )
        else:  # TODO: Remove this else
            text_ids, special_tokens_mask = clip_kc_new.tokenize(
                batch[f"text"], truncate=True, return_special_tokens_mask=True
            )
            text_ids = text_ids.to(self.device)
            special_tokens_mask = special_tokens_mask.to(self.device)

        if mask_text:  # ! This messes with text_feats and text_all_feats
            masked_text_ids, text_labels, text_masked_indices = self.mask_text_ids(
                text_ids, special_tokens_mask
            )
            # [N, C], [N, L, C]
            text_feats, text_all_feats = self.clip.encode_text(
                masked_text_ids,
                masked_indices=text_masked_indices,
                return_all_feats=True,
            )
        else:
            text_feats, text_all_feats = self.clip.encode_text(
                text_ids, return_all_feats=True
            )

        # Encode Video ########################################################
        video = batch[imgkey][0]
        if self.clip_type in ["ori", "evl", "kc", "kc_new"]:
            # [N, T, C, H, W] -> [N, C, T, H, W]
            video = video.contiguous().transpose(1, 2)

        # TODO: Remove this if
        # [N, C], [L, N, T, C]
        # video_feats for contrastive, video_all_feats for mlm, caption
        if mask_video:
            visual_masked_indices = self.mask_visual(video, mode=mode)
            video_feats, video_all_feats = self.clip.encode_video(
                video,
                return_all_feats=True,
                masked_indices=visual_masked_indices,
                mode=mode,
            )
        else:
            video_feats, video_all_feats = self.clip.encode_video(
                video, return_all_feats=True, mode=mode
            )

        ret = {
            "video": video,  # N, C, T, H, W
            "text_feats": text_feats,  # N, C
            "video_feats": video_feats,  # N, C
            "text_ids": text_ids,  # N, L
            "special_tokens_mask": special_tokens_mask,  # N, L
        }

        if self.mmt:
            # compute key features
            with torch.no_grad():  # no gradient to keys
                self._momentum_update_key_encoder()  # update the key encoder
                assert not any([mask_text, mask_video, mode != "video"])
                # TODO: We have BN, batch shuffle ?
                text_feats_k = self.clip_k.encode_text(text_ids, return_all_feats=False)
                # img, idx_unshuffle = batch_shuffle_ddp(img)
                video_feats_k = self.clip_k.encode_video(
                    video, return_all_feats=False, mode=mode
                )
                # video_feats_k = batch_unshuffle_ddp(video_feats_k, idx_unshuffle)
                ret.update(
                    {
                        "text_feats_k": text_feats_k,
                        "video_feats_k": video_feats_k,
                    }
                )

        # Mask Text Decoder ##################################################
        if mask_text:
            text_con_feats = text_feats
            text_feats = self.text_decoder(text_all_feats, video_all_feats)
            ret.update(
                {
                    # ! Modified the original, no other loss should do the same
                    "text_feats": text_feats,  # N, C
                    "text_labels": text_labels,  # N, L
                    "text_contrastive_feats": text_con_feats,  # N, L
                }
            )

        # Mask Visual Decoder#################################################
        if mask_video and hasattr(self, "visual_decoder"):
            mim_video_feats = self.visual_decoder(
                video_all_feats, visual_masked_indices
            )
            ret.update(
                {
                    "mim_video_feats": mim_video_feats,  # N, L, C
                    "visual_masked_indices": visual_masked_indices,  # N, L
                }
            )

        # Caption decoder   ##################################################
        if caption:
            cap_logits = self.caption_decoder(video_all_feats, text_all_feats[:, :-1])
            ret.update(
                {
                    "cap_logits": cap_logits,
                }
            )

        return ret

    def sanity_check(self):
        image = (
            self.clip_preprocess(
                Image.open(
                    "/mnt/petrelfs/liyizhuo/projects/all-in-one-cotrain/CoTraining/dog.png"
                )
            )
            .unsqueeze(0)
            .to(self.device)
        )
        T = 16
        B, C, H, W = image.shape
        video = image.repeat(T, 1, 1, 1).reshape(T, B, C, H, W).permute(1, 0, 2, 3, 4)
        self.eval()
        infer = self.infer(
            {
                "video": [video],
                "text": ["a diagram", "a dog", "a cat"],
            },
            mode="image",
        )
        score = (
            self.clip.logit_scale.exp() * infer["video_feats"] @ infer["text_feats"].t()
        )
        assert False, (score.softmax(dim=-1), self.clip.logit_scale.item())

    def forward(self, batch, batch_idx=None, mode="video"):
        # self.sanity_check()
        with torch.no_grad():
            self.clip.logit_scale.clamp_(0, math.log(100))
        ret = dict()
        if len(self.current_tasks) == 0:
            ret.update(self.infer(batch, mode=mode))
            return ret

        if "contrastive" in self.current_tasks:
            mask_text = "mlm" in self.current_tasks
            mask_video = "mim" in self.current_tasks
            caption = "cap" in self.current_tasks
            if any([mask_text, mask_video, caption]):
                contrastive_ret, contrastive_infer = objectives.compute_contrastive(
                    self,
                    batch,
                    return_infer=True,
                    mask_text=mask_text,
                    mask_video=mask_video,
                    caption=caption,
                    mode=mode,
                )
                ret.update(contrastive_ret)
            else:
                ret.update(objectives.compute_contrastive(self, batch, mode=mode))

        if "multiple_choice" in self.current_tasks:
            ret.update(objectives.compute_multiple_choice(self, batch))

        if "openend_vqa" in self.current_tasks:
            ret.update(objectives.compute_openend_vqa(self, batch))

        if "mlm" in self.current_tasks:
            if "contrastive" in self.current_tasks:  # Skip infer
                ret.update(objectives.compute_mlm(self, batch, infer=contrastive_infer))
            else:
                ret.update(objectives.compute_mlm(self, batch))

        if "mim" in self.current_tasks and hasattr(self, "visual_decoder"):
            if "contrastive" in self.current_tasks:  # Skip infer
                ret.update(
                    objectives.compute_mim(
                        self, batch, infer=contrastive_infer, mode=mode
                    )
                )
            else:
                ret.update(objectives.compute_mim(self, batch))

        if "cap" in self.current_tasks:
            if "contrastive" in self.current_tasks:  # Skip infer
                ret.update(
                    objectives.compute_cap(
                        self, batch, infer=contrastive_infer, mode=mode
                    )
                )
            else:
                ret.update(objectives.compute_cap(self, batch, mode=mode))

        if "zs_classify" in self.current_tasks:
            if self.text_ret is None:
                # print(f"Generate text features for in batch-{batch_idx}")
                self.text_ret = self.forward_text()
            ret.update(objectives.compute_zs_classify(self, batch, self.text_ret))

        return ret

    def forward_text(self):
        classes, num_text_aug, _ = text_prompt(prompt_type=self.prompt_type)
        text_inputs = classes.to(self.device)
        text_feats = self.clip.encode_text(text_inputs)
        # text_feats /= text_feats.norm(dim=-1, keepdim=True)

        ret = {
            "text_feats": text_feats,  # num_text_aug * num_classes, C
            "num_text_aug": num_text_aug,
        }
        return ret

    def forward_video(self, batch):
        img = batch["video"][0]
        if self.clip_type in ["ori", "evl", "kc", "kc_new"]:
            # [B, T, C, H, W] -> [B, C, T, H, W]
            img = img.contiguous().transpose(1, 2)
        video_feats = self.clip.encode_video(img)

        ret = {
            "video_feats": video_feats,  # N, C
        }
        return ret

    def training_step(self, batch, batch_idx):
        # gradually_freeze_by_layer(self, self.global_step, self.grad_unfreeze_int)
        cotrain_utils.set_task(self)
        # self.momentum_checkpoint()
        # co-training
        if "v" in batch and "i" in batch:
            video_output, image_output = {}, {}
            if not self.alt_data or batch_idx % 2 == 0:
                video_output = self(batch["v"], mode="video")
            if not self.alt_data or batch_idx % 2 == 1:
                image_output = self(batch["i"], mode="image")
            total_loss = sum([v for k, v in video_output.items() if "loss" in k]) + sum(
                [v for k, v in image_output.items() if "loss" in k]
            )
        else:
            output = self(batch, mode="video")
            total_loss = sum([v for k, v in output.items() if "loss" in k])
        return total_loss

    def training_epoch_end(self, outs):
        cotrain_utils.epoch_wrapup(self)

    def validation_step(self, batch, batch_idx):
        cotrain_utils.set_task(self)
        if "v" in batch and "i" in batch:
            video_output = self(batch["v"], mode="video")
            image_output = self(batch["i"], mode="image")
        else:
            output = self(batch, mode="video")

    def validation_epoch_end(self, outs):
        cotrain_utils.epoch_wrapup(self)
        self.text_ret = None

    def test_step(self, batch, batch_idx):
        cotrain_utils.set_task(self)
        if "v" in batch and "i" in batch:
            video_output = self(batch["v"], mode="video")
            image_output = self(batch["i"], mode="image")
        else:
            output = self(batch, mode="video")
        ret = dict()

        if self.hparams.config["loss_names"]["vqa"] > 0:
            ret.update(objectives.vqa_test_step(self, batch, image_output))

        return ret

    def test_epoch_end(self, outs):
        if isinstance(self.hparams.config["load_path"], str):
            model_name = self.hparams.config["load_path"].split("/")[-1][:-5]
        else:
            model_name = "multiple"

        if self.hparams.config["loss_names"]["vqa"] > 0:
            objectives.vqa_test_wrapup(outs, model_name)
        cotrain_utils.epoch_wrapup(self)

    def configure_optimizers(self):
        return cotrain_utils.set_schedule(self)
