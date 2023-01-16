import torch
import torch.nn as nn
import pytorch_lightning as pl
from transformers.models.bert.modeling_bert import BertConfig, BertEmbeddings
from CoTrain.modules import heads, cotrain_utils
from CoTrain.modules import objectives as objectives
from CoTrain.modules import base_vision_transformer as vit
from CoTrain.modules.temporal_roll import TemporalRoll
import torch.nn.functional as F
import math
from CoTrain.modules.cotrain_utils import state_dict_dino_fix


class VCOPHeader(torch.nn.Module):
    def __init__(self, tuple_len=3, feature_size=768):
        """
        In the constructor we instantiate two nn.Linear modules and assign them as
        member variables.
        """
        super(VCOPHeader, self).__init__()
        self.feature_size = feature_size
        self.fc7 = nn.Linear(self.feature_size * 2, 512)
        self.tuple_len = tuple_len
        pair_num = int(tuple_len * (tuple_len - 1) / 2)
        self.class_num = math.factorial(tuple_len)
        self.fc8 = nn.Linear(512 * pair_num, self.class_num)
        self.dropout = nn.Dropout(p=0.5)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        """
        """
        pf = []  # pairwise concat
        for i in range(self.tuple_len):
            for j in range(i + 1, self.tuple_len):
                pf.append(torch.cat([x[:, i], x[:, j]], dim=1))
        pf = [self.fc7(i) for i in pf]
        pf = [self.relu(i) for i in pf]
        h = torch.cat(pf, dim=1)
        h = self.dropout(h)
        h = self.fc8(h)  # logits
        return h


class CoTrainTransformerSS(pl.LightningModule):
    def __init__(self, config):
        super().__init__()
        self.save_hyperparameters()

        bert_config = BertConfig(
            vocab_size=config["vocab_size"],
            hidden_size=config["hidden_size"],
            num_hidden_layers=config["num_layers"],
            num_attention_heads=config["num_heads"],
            intermediate_size=config["hidden_size"] * config["mlp_ratio"],
            max_position_embeddings=config["max_text_len"],
            hidden_dropout_prob=config["drop_rate"],
            attention_probs_dropout_prob=config["drop_rate"],
        )

        self.text_embeddings = BertEmbeddings(bert_config)
        self.text_embeddings.apply(objectives.init_weights)

        self.token_type_embeddings = nn.Embedding(2, config["hidden_size"])
        self.token_type_embeddings.apply(objectives.init_weights)

        flag = 0
        if self.hparams.config["load_path"] == "":
            while not flag == 1:
                try:
                    self.transformer = getattr(vit, self.hparams.config["vit"])(
                        pretrained=True, config=self.hparams.config
                    )
                    flag = 1
                except:
                    print("load pretrained failed, try again")
                    flag = 0
        else:
            self.transformer = getattr(vit, self.hparams.config["vit"])(
                pretrained=False, config=self.hparams.config
            )

        self.pooler = heads.Pooler(config["hidden_size"])
        self.pooler.apply(objectives.init_weights)

        # num frames
        self.num_frames = config["num_frames"]  # a global variable to identify if image/video

        if config["loss_names"]["mlm"] > 0:
            self.mlm_score = heads.MLMHead(bert_config)
            self.mlm_score.apply(objectives.init_weights)

        if config["loss_names"]["vtm"] > 0:
            self.vtm_score = heads.vtmHead(config["hidden_size"])
            self.vtm_score.apply(objectives.init_weights)

        if config["loss_names"]["mpp"] > 0:
            self.mpp_score = heads.MPPHead(bert_config)
            self.mpp_score.apply(objectives.init_weights)

        # vtc may also used for pretrain
        # == for video text contrastive learning
        if config["loss_names"]["vtc"] > 0:
            print("initalize video project and txt projection")
            # v1
            self.txt_proj = nn.Linear(config["hidden_size"], config["shared_embedding_dim"])
            self.vid_proj = nn.Linear(config["hidden_size"], config["shared_embedding_dim"])
            # v2
            # self.vid_proj = nn.Sequential(
            #     nn.Dropout(0.5),
            #     nn.Linear(config["hidden_size"], config["hidden_size"] // 2),
            #     nn.LayerNorm(config["hidden_size"] // 2),
            #     nn.GELU(),
            #     nn.Linear(config["hidden_size"] // 2, config["shared_embedding_dim"]),
            # )
            # self.txt_proj = nn.Sequential(
            #     nn.Dropout(0.5),
            #     nn.Linear(config["hidden_size"], config["hidden_size"] // 2),
            #     nn.LayerNorm(config["hidden_size"] // 2),
            #     nn.GELU(),
            #     nn.Linear(config["hidden_size"] // 2, config["shared_embedding_dim"]),
            # )
            self.txt_proj.apply(objectives.init_weights)
            self.vid_proj.apply(objectives.init_weights)
        # == end
        if (
            self.hparams.config["load_path"] != ""
            and not self.hparams.config["test_only"]
        ):
            print("0" * 200)
            ckpt = torch.load(self.hparams.config["load_path"], map_location="cpu")
            state_dict = ckpt["state_dict"]
            # if downstream max text token length not consistent with pretrain
            if self.text_embeddings.position_embeddings.weight.size() != state_dict['text_embeddings.position_embeddings.weight'].size():
                state_dict.pop('text_embeddings.position_embeddings.weight', None)
                state_dict.pop('text_embeddings.position_ids', None)
            new_state_dict = state_dict_dino_fix(state_dict, self.state_dict())
            # new_state_dict = self._inflate_positional_embeds(state_dict)
            self.load_state_dict(new_state_dict, strict=False)
            # self.load_state_dict(state_dict, strict=False)
        if self.hparams.config["linear_evaluation"]:
            for name, param in self.named_parameters():
                # only train project layer
                if 'mlm_score' in name or 'vtm_score' in name or 'mpp_score' in name:
                    param.requires_grad = True
                elif 'txt_proj' in name or 'vid_proj' in name:
                    param.requires_grad = True
                elif 'pooler' in name:
                    param.requires_grad = True
                else:
                    param.requires_grad = False
            # flag = False
            # for name, param in self.named_parameters():
            #     if '20' in name:
            #         flag = True
            #     param.requires_grad = flag
        # trainable_params = filter(lambda p: p.requires_grad, self.parameters())
        # ===================== Downstream ===================== #

        hs = self.hparams.config["hidden_size"]
        # print(config["loss_names"])
        if config["loss_names"]["multiple_choice"] > 0:
            self.vtm_score = heads.vtmHead(config["hidden_size"])
            self.vtm_score.apply(objectives.init_weights)

        # alex:  vcr q2a task
        if config["loss_names"]["vcr_q2a"] > 0:
            self.vtm_score = heads.vtmHead(config["hidden_size"])
            self.vtm_score.apply(objectives.init_weights)

        # alex:  tvqa
        if config["loss_names"]["mc_vqa"] > 0:
            self.vtm_score = heads.vtmHead(config["hidden_size"])
            self.vtm_score.apply(objectives.init_weights)

        if self.hparams.config["loss_names"]["vqa"] > 0:
            vs = self.hparams.config["vqav2_label_size"]
            self.vqa_classifier = nn.Sequential(
                nn.Linear(hs, hs * 2),
                nn.LayerNorm(hs * 2),
                nn.GELU(),
                nn.Linear(hs * 2, vs),
            )
            self.vqa_classifier.apply(objectives.init_weights)

        # alex: add for vcr: q2a
        if self.hparams.config["loss_names"]["vcr_q2a"] > 0:
            # for q2a
            self.rank_output = nn.Linear(hs, 1)
            self.rank_output.weight.data = self.vtm_score.fc.weight.data[1:, :]
            self.rank_output.bias.data = self.vtm_score.fc.bias.data[1:]
            # for qa2r
            self.rank_output_2 = nn.Linear(hs, 1)
            self.rank_output_2.weight.data = self.vtm_score.fc.weight.data[1:, :]
            self.rank_output_2.bias.data = self.vtm_score.fc.bias.data[1:]

            self.margin = 0.2

        # add for vcop prediction
        if self.hparams.config["loss_names"]["vcop"] > 0:
            self.vcop_classifier = VCOPHeader(tuple_len=self.num_frames, feature_size=hs)

        # add for tvqa
        if self.hparams.config["loss_names"]["mc_vqa"] > 0:
            # # v1: for q2a with vtm_score
            # self.rank_output = nn.Linear(hs, 1)
            # self.rank_output.weight.data = self.vtm_score.fc.weight.data[1:, :]
            # self.rank_output.bias.data = self.vtm_score.fc.bias.data[1:]
            # self.dropout = nn.Dropout(0.1)
            self.mc_vqa_classifier = nn.Sequential(
                nn.Dropout(0.1),
                nn.Linear(hs, 256),
                nn.LayerNorm(256),
                nn.GELU(),
                nn.Linear(256, 1),
            )
            self.mc_vqa_classifier.apply(objectives.init_weights)

        # alex: add for openend_vqa
        if self.hparams.config["loss_names"]["openend_vqa"] > 0:
            vs = self.hparams.config["msrvttqa_label_size"]
            # self.vqa_classifier = nn.Sequential(
            #     nn.Dropout(0.5),
            #     nn.Linear(hs, hs * 2),
            #     nn.LayerNorm(hs * 2),
            #     nn.GELU(),
            #     nn.Linear(hs * 2, vs),
            # )
            # small dataset
            self.vqa_classifier = nn.Sequential(
                nn.Dropout(0.5),
                nn.Linear(hs, hs//2),
                nn.LayerNorm(hs//2),
                nn.GELU(),
                nn.Linear(hs//2, vs),
            )
            self.vqa_classifier.apply(objectives.init_weights)

        if self.hparams.config["loss_names"]["nlvr2"] > 0:
            self.nlvr2_classifier = nn.Sequential(
                nn.Linear(hs * 2, hs * 2),
                nn.LayerNorm(hs * 2),
                nn.GELU(),
                nn.Linear(hs * 2, 2),
            )
            self.nlvr2_classifier.apply(objectives.init_weights)
            emb_data = self.token_type_embeddings.weight.data
            self.token_type_embeddings = nn.Embedding(3, hs)
            self.token_type_embeddings.apply(objectives.init_weights)
            self.token_type_embeddings.weight.data[0, :] = emb_data[0, :]
            self.token_type_embeddings.weight.data[1, :] = emb_data[1, :]
            self.token_type_embeddings.weight.data[2, :] = emb_data[1, :]

        if self.hparams.config["loss_names"]["irtr"] > 0:
            self.rank_output = nn.Linear(hs, 1)
            self.rank_output.weight.data = self.vtm_score.fc.weight.data[1:, :]
            self.rank_output.bias.data = self.vtm_score.fc.bias.data[1:]
            self.margin = 0.2
            # for p in self.vtm_score.parameters(): # alex: requires_grad = true?
            #     p.requires_grad = False

        # test msrvtt multiple choice without finetune
        if self.hparams.config["loss_names"]["multiple_choice"] > 0:
            self.rank_output = nn.Linear(hs, 1)
            self.rank_output.weight.data = self.vtm_score.fc.weight.data[1:, :]
            self.rank_output.bias.data = self.vtm_score.fc.bias.data[1:]
            self.margin = 0.2

        cotrain_utils.set_metrics(self)
        self.current_tasks = list()

        self.temporal_roll_module = TemporalRoll(n_segment=self.num_frames, v=0)

        # ===================== load downstream (test_only) ======================

        if self.hparams.config["load_path"] != "" and self.hparams.config["test_only"]:
            print("====load checkpoint=====")
            ckpt = torch.load(self.hparams.config["load_path"], map_location="cpu")
            state_dict = ckpt["state_dict"]
            print('*' * 30)
            print("current state dict")
            print('*' * 30)
            for k, v in self.state_dict().items():
                print(k)
            # temporal embed and fix model ?
            # new_state_dict = state_dict_data_parallel_fix(state_dict, self.state_dict())
            new_state_dict = state_dict_dino_fix(state_dict, self.state_dict())
            # new_state_dict = self._inflate_positional_embeds(state_dict)
            self.load_state_dict(new_state_dict, strict=False)
            # self.load_state_dict(state_dict, strict=False)
        # # # print learnable param
        # for name, param in self.named_parameters():
        #     if param.requires_grad:
        #         print("learned param: ", name)

    def infer(
        self,
        batch,
        mask_text=False,
        mask_video=False,
        video_token_type_idx=1,
        video_embeds=None,
        video_masks=None,
        input_video_only=False,
        input_text_only=False,
        mode="video"
    ):
        # if text: process in normal video
        # if video: repeat the text tensor for K times
        if f"video_{video_token_type_idx - 1}" in batch:
            imgkey = f"video_{video_token_type_idx - 1}"
        else:
            imgkey = "video"
        do_mlm = "_mlm" if mask_text else ""
        text_ids = batch[f"text_ids{do_mlm}"]
        text_labels = batch[f"text_labels{do_mlm}"]
        text_masks = batch[f"text_masks"]
        # print(batch[imgkey])
        self.num_frames = batch[imgkey][0].size(1)
        if not input_video_only:
            text_embeds = self.text_embeddings(text_ids)
            video_labels = None
            patch_index = None
        if not input_text_only:
            if video_embeds is None and video_masks is None:
                img = batch[imgkey][0]
                img = img.contiguous().view(-1, img.size()[2], img.size()[3], img.size()[4])  # btchw to [bt]chw
                (
                    video_embeds,
                    video_masks,
                    patch_index,
                    video_labels,
                ) = self.transformer.visual_embed(
                    img,
                    max_image_len=self.hparams.config["max_image_len"],
                    mask_it=mask_video,
                    mode=mode
                )
            else:
                patch_index, video_labels = (
                    None,
                    None,
                )
        if not input_video_only:
            text_embeds = text_embeds + self.token_type_embeddings(torch.zeros_like(text_masks))
            text_embeds = torch.repeat_interleave(text_embeds, self.num_frames, dim=0)
            text_masks = torch.repeat_interleave(text_masks, self.num_frames, dim=0)
        if not input_text_only:
            video_embeds = video_embeds + self.token_type_embeddings(torch.full_like(video_masks, video_token_type_idx))

        # print(text_embeds.size(), video_embeds.size())
        if not input_text_only and not input_video_only:
            co_embeds = torch.cat([text_embeds, video_embeds], dim=1)
            co_masks = torch.cat([text_masks, video_masks], dim=1)
            x = co_embeds
        if input_text_only:
            x = text_embeds
            co_masks = text_masks
        if input_video_only:
            x = video_embeds
            co_masks = video_masks

        for i, blk in enumerate(self.transformer.blocks):
            # perform temporal roll operation for temporal modeling [video only]
            if self.num_frames > 1 and not input_video_only and not input_text_only:
                text_feats, image_feats = (
                    x[:, : text_embeds.shape[1]],
                    x[:, text_embeds.shape[1]:],
                )
                image_feats = self.temporal_roll_module(image_feats, i)
                x = torch.cat((text_feats, image_feats), dim=1)
            x, _attn = blk(x, mask=co_masks)
        x = self.transformer.norm(x)
        # reshape to video tensor
        x = x.view(x.size(0) // self.num_frames, -1, x.size(-2),
                                     x.size(-1))
        # add vcop here
        h = None
        if self.hparams.config["loss_names"]["vcop"] > 0 and mode == "video":
            h = x
        x = torch.mean(x, dim=1)
        if input_text_only:
            text_feats = x
            if "vtc" in self.current_tasks:
                text_feats = self.txt_proj(text_feats)
            video_feats = None
        if input_video_only:
            video_feats = x
            if "vtc" in self.current_tasks:
                video_feats = self.vid_proj(video_feats)
            text_feats = None
        if not input_text_only and not input_video_only:
            text_feats, video_feats = (
                x[:, : text_embeds.shape[1]],
                x[:, text_embeds.shape[1]:],
            )
        cls_feats = self.pooler(x)
        if not input_video_only:
            text_masks = text_masks[::self.num_frames].contiguous()
        if not input_text_only:
            video_masks = video_masks[::self.num_frames].contiguous()
        ret = {
            "text_feats": text_feats,
            "video_feats": video_feats,
            "cls_feats": cls_feats,
            "raw_cls_feats": x[:, 0],
            "video_labels": video_labels,
            "video_masks": video_masks,
            "text_labels": text_labels,
            "text_ids": text_ids,
            "text_masks": text_masks,
            "patch_index": patch_index,
            "vcop_features": h
        }
        return ret

    def forward(self, batch, mode="video"):
        ret = dict()
        if len(self.current_tasks) == 0:
            ret.update(self.infer(batch, mode=mode))
            return ret

        # Masked Language Modeling
        if "mlm" in self.current_tasks:
            ret.update(objectives.compute_mlm(self, batch, mode=mode))

        # video Text Matching
        if "vtm" in self.current_tasks:
            ret.update(objectives.compute_vtm_wpa(self, batch, mode=mode))
            # ret.update(objectives.compute_vtm_wpa_dino(self, batch, mode=mode))

        # video Text Contrastive
        if "vtc" in self.current_tasks:
            ret.update(objectives.compute_vtc(self, batch, mode=mode))

        # Visual Question Answering
        if "vqa" in self.current_tasks:
            ret.update(objectives.compute_vqa(self, batch))

        # alex: msrvtt Visual Question Answering
        if "openend_vqa" in self.current_tasks:
            ret.update(objectives.compute_openend_vqa(self, batch))

        # alex: vcop only for video
        if "vcop" in self.current_tasks and mode == "video":
            ret.update(objectives.compute_vcop(self, batch))

        # alex: vcr qa
        if "vcr_q2a" in self.current_tasks:
            ret.update(objectives.compute_vcr_q2a(self, batch))

        # alex: mc_vqa
        if "mc_vqa" in self.current_tasks:
            ret.update(objectives.compute_mc_vqa_q2a(self, batch))

        # alex: msrvtt multiple choice setting
        if "multiple_choice" in self.current_tasks:
            ret.update(objectives.compute_multiple_choice(self, batch))

        # video Retrieval and Text Retrieval
        if "irtr" in self.current_tasks:
            ret.update(objectives.compute_irtr(self, batch))

        return ret

    def training_step(self, batch, batch_idx):
        cotrain_utils.set_task(self)
        # co-training
        if "v" in batch and "i" in batch:
            video_output = self(batch["v"], mode="video")
            image_output = self(batch["i"], mode="image")
            total_loss = sum([v for k, v in video_output.items() if "loss" in k]) + sum([v for k, v in image_output.items() if "loss" in k])
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
        model_name = self.hparams.config["load_path"].split("/")[-1][:-5]

        if self.hparams.config["loss_names"]["vqa"] > 0:
            objectives.vqa_test_wrapup(outs, model_name)
        cotrain_utils.epoch_wrapup(self)

    def configure_optimizers(self):
        return cotrain_utils.set_schedule(self)

    def _inflate_positional_embeds(self, new_state_dict, load_temporal_fix='zeros'):
        # allow loading of timesformer with fewer num_frames
        curr_keys = list(self.state_dict().keys())
        if 'transformer.temporal_embed' in new_state_dict and 'transformer.temporal_embed' in curr_keys:
            load_temporal_embed = new_state_dict['transformer.temporal_embed']
            load_num_frames = load_temporal_embed.shape[1]
            curr_num_frames = self.hparams.config['num_frames']
            embed_dim = load_temporal_embed.shape[2]

            if load_num_frames != curr_num_frames:
                if load_num_frames > curr_num_frames:
                    print(f'### loaded {self.hparams.config["vit"]} model has MORE frames than current...'
                          f'### loading weights, filling in the extras via {load_temporal_fix}')
                    new_temporal_embed = load_temporal_embed[:, :curr_num_frames, :]
                else:
                    print(f'### loaded {self.hparams.config["vit"]} model has FEWER frames than current...'
                          f'### loading weights, filling in the extras via {load_temporal_fix}')
                    if load_temporal_fix == 'zeros':
                        new_temporal_embed = torch.zeros([load_temporal_embed.shape[0], curr_num_frames, embed_dim])
                        new_temporal_embed[:, :load_num_frames] = load_temporal_embed
                    elif load_temporal_fix in ['interp', 'bilinear']:
                        # interpolate
                        # unsqueeze so pytorch thinks its an image
                        mode = 'nearest'
                        if load_temporal_fix == 'bilinear':
                            mode = 'bilinear'
                        load_temporal_embed = load_temporal_embed.unsqueeze(0)
                        new_temporal_embed = F.interpolate(load_temporal_embed,
                                                           (curr_num_frames, embed_dim), mode=mode).squeeze(0)
                    else:
                        raise NotImplementedError
                new_state_dict['transformer.temporal_embed'] = new_temporal_embed
        # allow loading with smaller spatial patches. assumes custom border crop, to append the
        # border patches to the input sequence
        if 'transformer.pos_embed' in new_state_dict and 'transformer.pos_embed' in curr_keys:
            load_pos_embed = new_state_dict['transformer.pos_embed']
            load_num_patches = load_pos_embed.shape[1]
            curr_pos_embed = self.state_dict()['transformer.pos_embed']
            if load_num_patches != curr_pos_embed.shape[1]:
                raise NotImplementedError(
                    'Loading models with different spatial resolution / patch number not yet implemented, sorry.')

        return new_state_dict