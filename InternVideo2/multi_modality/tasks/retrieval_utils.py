import datetime
import logging
import time

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
from einops import rearrange

from models.criterions import get_sim
from utils.basic_utils import MetricLogger
from utils.distributed import get_rank, get_world_size

logger = logging.getLogger(__name__)


def extract_text_feats(texts, max_txt_l, tokenizer, model, device, return_ids=False):
    num_text = len(texts)
    text_bs = 256
    text_feats = []
    text_atts = []
    
    if return_ids:
        text_ids = []

    for i in range(0, num_text, text_bs):
        text = texts[i : min(num_text, i + text_bs)]
        text_input = tokenizer(
            text,
            padding="max_length",
            truncation=True,
            max_length=max_txt_l,
            return_tensors="pt",
        ).to(device) # NOTE not need to cast

        text_feat = model.encode_text(text_input)[0]
        text_feats.append(text_feat)
        text_atts.append(text_input.attention_mask)
        if return_ids:
            text_ids.append(text_input.input_ids)

    text_feats = torch.cat(text_feats, dim=0)
    text_atts = torch.cat(text_atts, dim=0)
    if return_ids:
        text_ids = torch.cat(text_ids, dim=0)
        return text_feats, text_atts, text_ids
    else:
        return text_feats, text_atts

def extract_vision_feats(data_loader, model, device, config):
    if config.use_half_precision: 
        if config.get('use_bf16', False):
            cast_dtype = torch.bfloat16
        else:
            cast_dtype = torch.float16
    else:
        cast_dtype = None

    image_feats_all = []
    pooled_image_feats_all = []
    metric_logger = MetricLogger(delimiter="  ")
    header = "extracting image feats"
    iterator = metric_logger.log_every(data_loader, 100, header)
    for image, img_id in iterator:
        image = image.to(device, dtype=cast_dtype, non_blocking=True)
        image_feat, pooled_image_feat = model.encode_vision(image, test=True)
        if len(pooled_image_feat.shape) == 2:
            pooled_image_feat = pooled_image_feat.unsqueeze(1)  # make av_fusion happy
        if config.evaluation.eval_frame_ensemble == "concat": 
            if len(image_feat.shape) == 4:
                image_feat = rearrange(image_feat, "b t l c -> b (t l) c").contiguous()
            image_feat = image_feat.unsqueeze(1)  # (bsz, 1, #frm*L, d)
        else:
            assert config.video_input.num_frames == 1, "only support single-frame"
            assert config.evaluation.eval_frame_ensemble in ["mean", "max", "lse"]
        if config.evaluation.eval_offload:
            image_feats_all.append(image_feat.cpu())
            pooled_image_feats_all.append(pooled_image_feat.cpu())
        else:
            image_feats_all.append(image_feat)
            pooled_image_feats_all.append(pooled_image_feat)

    image_feats_all = torch.cat(image_feats_all, dim=0)
    pooled_image_feats_all = torch.cat(pooled_image_feats_all, dim=0)

    return image_feats_all, pooled_image_feats_all

def extract_audio_feats(data_loader, model, device, config):
    if config.use_half_precision: 
        if config.get('use_bf16', False):
            cast_dtype = torch.bfloat16
        else:
            cast_dtype = torch.float16
    else:
        cast_dtype = None

    audio_feats_all = []
    pooled_audio_feats_all = []
    metric_logger = MetricLogger(delimiter="  ")
    header = "extracting audio feats"
    iterator = metric_logger.log_every(data_loader, 100, header)
    for audio, _ in iterator:
        audio = audio.to(device, dtype=cast_dtype, non_blocking=True)
        audio_feat, pooled_audio_feat = model.encode_audio(audio, test=True)
        audio_feat = audio_feat.unsqueeze(1)  # make deep_fusion happy
        pooled_audio_feat = pooled_audio_feat.unsqueeze(1)
        if config.evaluation.eval_offload:
            audio_feats_all.append(audio_feat.cpu())
            pooled_audio_feats_all.append(pooled_audio_feat.cpu())
        else:
            audio_feats_all.append(audio_feat)
            pooled_audio_feats_all.append(pooled_audio_feat)

    audio_feats_all = torch.cat(audio_feats_all, dim=0)

    pooled_audio_feats_all = torch.cat(pooled_audio_feats_all, dim=0)
    return audio_feats_all, pooled_audio_feats_all

def extract_audio_vision_feats(data_loader, model, device, config):
    if config.use_half_precision: 
        if config.get('use_bf16', False):
            cast_dtype = torch.bfloat16
        else:
            cast_dtype = torch.float16
    else:
        cast_dtype = None

    audio_feats_all = []
    pooled_audio_feats_all = []
    image_feats_all = []
    pooled_image_feats_all = []
    metric_logger = MetricLogger(delimiter="  ")
    header = "extracting audio and vision feats"
    iterator = metric_logger.log_every(data_loader, 100, header)
    for media, _ in iterator:
        audio = media[0]
        image = media[1]
        audio = audio.to(device, dtype=cast_dtype, non_blocking=True)
        image = image.to(device, dtype=cast_dtype, non_blocking=True)
        audio_feat, pooled_audio_feat = model.encode_audio(audio, test=True)
        audio_feat = audio_feat.unsqueeze(1)  # make deep_fusion happy
        pooled_audio_feat = pooled_audio_feat.unsqueeze(1)
        image_feat, pooled_image_feat = model.encode_vision(image, test=True)
        if len(pooled_image_feat.shape) == 2:
            pooled_image_feat = pooled_image_feat.unsqueeze(1)  # make av_fusion happy
        if config.evaluation.eval_frame_ensemble == "concat":
            if len(image_feat.shape) == 4:
                image_feat = rearrange(image_feat, "b t l c -> b (t l) c").contiguous()
            image_feat = image_feat.unsqueeze(1)  # (bsz, 1, #frm*L, d)
        else:
            assert config.video_input.num_frames == 1, "only support single-frame"
            assert config.evaluation.eval_frame_ensemble in ["mean", "max", "lse"]
        if config.evaluation.eval_offload:
            audio_feats_all.append(audio_feat.cpu())
            pooled_audio_feats_all.append(pooled_audio_feat.cpu())
            image_feats_all.append(image_feat.cpu())
            pooled_image_feats_all.append(pooled_image_feat.cpu())
        else:
            audio_feats_all.append(audio_feat)
            pooled_audio_feats_all.append(pooled_audio_feat)
            image_feats_all.append(image_feat)
            pooled_image_feats_all.append(pooled_image_feat)

    audio_feats_all = torch.cat(audio_feats_all, dim=0)
    pooled_audio_feats_all = torch.cat(pooled_audio_feats_all, dim=0)
    image_feats_all = torch.cat(image_feats_all, dim=0)
    pooled_image_feats_all = torch.cat(pooled_image_feats_all, dim=0)

    return audio_feats_all, pooled_audio_feats_all, image_feats_all, pooled_image_feats_all


@torch.no_grad()
def evaluation_wrapper(model, data_loader, tokenizer, device, config, prefix=""):
    amp_eval_enabled = config.use_half_precision and not (hasattr(config, "deepspeed") and config.deepspeed.enable)
    logger.info(f"Begin to eval, model_without_ddp.dtype={model.dtype if hasattr(model, 'dtype')  else None}, amp_eval_enabled={amp_eval_enabled}, dtype={torch.bfloat16 if config.get('use_bf16', False) else torch.float16}")
    with torch.cuda.amp.autocast(enabled=amp_eval_enabled, dtype=torch.bfloat16 if config.get('use_bf16', False) else torch.float16):
        i2t_match, t2i_match = None, None
        if "qformer" in config.model.model_cls.lower():
            i2t_match, t2i_match, i2t_sim, t2i_sim, i2t_dsl, t2i_dsl = evaluation_qformer(
                model, data_loader, tokenizer, device, config
            )
        elif "blip" in config.model.model_cls.lower():
            raise NotImplementedError
        elif "clip" in config.model.model_cls.lower() or 'coca' in config.model.model_cls.lower():
            # raise NotImplementedError
            i2t_sim, t2i_sim, i2t_dsl, t2i_dsl = evaluation_clip(
                model, data_loader, tokenizer, device, config
            )
        else:
            i2t_match, t2i_match, i2t_sim, t2i_sim, i2t_dsl, t2i_dsl = evaluation(
                model, data_loader, tokenizer, device, config
            )

        if hasattr(data_loader.dataset, "num_prompts"):
            np = data_loader.dataset.num_prompts
            logger.info(f"Using {np} prompts, we need reshape and mean!!!")
            nt = len(data_loader.dataset.text) // np
            if i2t_match is not None:
                i2t_match = i2t_match.reshape((i2t_match.shape[0], nt, np)).mean(axis=-1)
                t2i_match = t2i_match.reshape((nt, np, t2i_match.shape[1])).mean(axis=1)
            i2t_sim = i2t_sim.reshape((i2t_sim.shape[0], nt, np)).mean(axis=-1)
            t2i_sim = t2i_sim.reshape((nt, np, t2i_sim.shape[1])).mean(axis=1)
            i2t_dsl = i2t_dsl.reshape((i2t_dsl.shape[0], nt, np)).mean(axis=-1)
            t2i_dsl = t2i_dsl.reshape((nt, np, t2i_dsl.shape[1])).mean(axis=1)

    score_pairs = [
        (prefix + "_sim", i2t_sim, t2i_sim),
        (prefix + "_dsl", i2t_dsl, t2i_dsl),
    ]
    if i2t_match is not None:
        if config.evaluation.get('use_dsl_for_match', False):
            score_pairs.append((prefix + "_match (use_dsl)", i2t_match, t2i_match))
        else:
            score_pairs.append((prefix + "_match", i2t_match, t2i_match))

    res = dict()
    for name, i2t, t2i in score_pairs:
        if i2t is not None:
            txt2img_ids = data_loader.dataset.txt2img
            img2txt_ids = data_loader.dataset.img2txt
            res[name] = itm_eval(i2t, t2i, txt2img_ids, img2txt_ids)
    return res


@torch.no_grad()
def evaluation(model, data_loader, tokenizer, device, config):
    model.eval()

    use_dsl_for_match = config.evaluation.get('use_dsl_for_match', False)

    metric_logger = MetricLogger(delimiter="  ")
    header = "Evaluation:"
    dtype = torch.half if config.use_half_precision else torch.float
    media_type = data_loader.dataset.media_type
    use_subtitle = hasattr(data_loader.dataset, "use_subtitle") and data_loader.dataset.use_subtitle
    if use_subtitle:
        assert media_type in ["video", "audio_video"], f"Not support media_type: {media_type}."
        assert hasattr(data_loader.dataset, "subtitle") and data_loader.dataset.subtitle is not None, "You don't have subtitle to use."

    logger.info(f"Start evaluation for media_type={media_type}")
    assert media_type in ['audio', 'video', 'audio_video'], f"Not implement evaluation of {media_type}"

    logger.info("Computing dual encoder features...")
    start_time = time.time()

    # this computes all features in each GPU
    texts = data_loader.dataset.text
    # max_txt_l of eval depends on data_cofig
    max_txt_l = data_loader.dataset.max_txt_l 

    text_feats, text_atts = extract_text_feats(
        texts, max_txt_l, tokenizer, model, device
    )  # (bsz, Lt, d), (bsz, Lt)

    if use_subtitle:
        subtitle_feats, _ = extract_text_feats(
            data_loader.dataset.subtitle, max_txt_l, tokenizer, model, device
        ) # (bsz, Lt, d), (bsz, Lt)
        subtitle_proj = model.text_proj(subtitle_feats[:, 0]).unsqueeze(1)
        subtitle_feats = subtitle_feats.unsqueeze(1) 
        
    if media_type == 'video':
        image_feats, pooled_image_feats = extract_vision_feats(
            data_loader, model, device, config
        )  # (bsz, 1, #frm*Li, d) or (bsz, #frm, Li, d), (bsz, #frm, d)
        logger.info("Finished vision feature extraction")
        logger.info("Computing ITC scores [dot-product]")
        if config.evaluation.eval_offload:
            # image_feats = image_feats.to(device, non_blocking=True) image_feats will cause OOM!!!
            pooled_image_feats = pooled_image_feats.to(device, non_blocking=True)

        if use_subtitle:
            # print(subtitle_proj.shape, pooled_image_feats.shape)
            i2t_scores, t2i_scores = get_sim(
                model.vs_fusion(torch.concat([subtitle_proj, model.vision_proj(pooled_image_feats)], dim=-1)), model.text_proj(text_feats[:, 0])
            )
        else:
            i2t_scores, t2i_scores = get_sim(
                model.vision_proj(pooled_image_feats), model.text_proj(text_feats[:, 0])
            )

        if use_dsl_for_match:
            logger.info("use_dsl_for_match!!!")
            old_i2t_scores, old_t2i_scores = i2t_scores, t2i_scores
            i2t_scores = old_i2t_scores * old_i2t_scores.softmax(dim=0)
            t2i_scores = old_i2t_scores.T * old_i2t_scores.T.softmax(dim=0)

        num_medias = len(data_loader.dataset.image)
        
        # pooled_media_feats = pooled_image_feats
        if use_subtitle:
            media_feats = torch.concat([subtitle_feats, image_feats], dim=-2)
            if hasattr(model, "vstm_head"):
                match_head = model.vstm_head
            else:
                match_head = None
        else:
            media_feats = image_feats
            if hasattr(model, "itm_head"):
                match_head = model.itm_head
            else:
                match_head = None

    elif media_type == 'audio':
        audio_feats, pooled_audio_feats = extract_audio_feats(
            data_loader, model, device, config
        )
        logger.info("Finished audio feature extraction")
        logger.info("Computing ITC scores [dot-product]")
        if config.evaluation.eval_offload:
            pooled_audio_feats = pooled_audio_feats.to(device, non_blocking=True)

        i2t_scores, t2i_scores = get_sim(
            model.audio_proj(pooled_audio_feats), model.text_proj(text_feats[:, 0])
        )

        num_medias = len(data_loader.dataset.audio)
        media_feats = audio_feats
        # pooled_media_feats = pooled_audio_feats
        if hasattr(model, "atm_head"):
            match_head = model.atm_head
        else:
            match_head = None

    elif media_type == 'audio_video':
        audio_feats, pooled_audio_feats, image_feats, pooled_image_feats = extract_audio_vision_feats(
            data_loader, model, device, config
        )
        logger.info("Finished audio and vision feature extraction")

        logger.info("Computing ITC scores [dot-product]")
        if config.evaluation.eval_offload:
            pooled_audio_feats = pooled_audio_feats.to(device, non_blocking=True)
            pooled_image_feats = pooled_image_feats.to(device, non_blocking=True)

        if use_subtitle:
            i2t_scores, t2i_scores = get_sim(
                model.avs_fusion(torch.concat([model.audio_proj(pooled_audio_feats), subtitle_proj, model.vision_proj(pooled_image_feats)], dim=-1)), model.text_proj(text_feats[:, 0])
            )
        else:
            i2t_scores, t2i_scores = get_sim(
                model.av_fusion(torch.concat([model.audio_proj(pooled_audio_feats), model.vision_proj(pooled_image_feats)], dim=-1)), model.text_proj(text_feats[:, 0])
            )

        num_medias = len(data_loader.dataset.image)
        if use_subtitle:
            media_feats = torch.concat([audio_feats, subtitle_feats, image_feats], dim=-2)
            # pooled_media_feats = pooled_audio_feats
            if hasattr(model, "avstm_head"):
                match_head = model.avstm_head
            else:
                match_head = None
        else:
            media_feats = torch.concat([audio_feats, image_feats], dim=-2)
            # pooled_media_feats = pooled_audio_feats
            if hasattr(model, "avtm_head"):
                match_head = model.avtm_head
            else:
                match_head = None
    else:
        raise NotImplementedError(media_type)
    
    logger.info("Computing ITC scores [dot-product], done!")
    
    if match_head is not None:
        i2t_scores_x = torch.full((num_medias, len(texts)), -100.0).to(
            device, torch.float, non_blocking=True
        )

        # computes only part of the scores at each GPU, gather at the end
        logger.info("Rerank dual-encoder results with cross-encoder...")
        num_tasks = get_world_size()
        rank = get_rank()
        # only uses the part associated with the raw eval set
        # compute media2text #
        step = num_medias // num_tasks + 1
        start = rank * step
        end = min(num_medias, start + step)

        text_encoder = model.get_text_encoder()
        iterator = metric_logger.log_every(i2t_scores[start:end], 100, header)
        logger.info(f"i2t_scores.shape {i2t_scores[start:end].shape}")

        # generate score for each clip, and aggregate all clip scores for a video
        n_clip_per_video = (
            media_feats.shape[1] if not config.deep_fusion else media_feats[0].shape[1]
        )

        assert not config.deep_fusion and n_clip_per_video == 1, f"Not implemented for config.deep_fusion={config.deep_fusion} n_clip_per_video={n_clip_per_video}"

        logger.info(
            f"n_clip_per_video={n_clip_per_video}, with eval_frame_ensemble={config.evaluation.eval_frame_ensemble}"
        )

        for i, sims in enumerate(iterator):
            k = min(len(sims), config.evaluation.k_test)
            topk_sim, topk_idx = sims.topk(k=k, dim=0)

            clip_scores = []
            for clip_idx in range(n_clip_per_video):
                if config.deep_fusion:
                    encoder_output = [
                        feat[start + i, clip_idx].to(device, non_blocking=True)
                        if config.evaluation.eval_offload
                        else feat[start + i, clip_idx]
                        for feat in media_feats
                    ]

                else:
                    encoder_output = (
                        media_feats[start + i, clip_idx].to(device, non_blocking=True)
                        if config.evaluation.eval_offload
                        else media_feats[start + i, clip_idx]
                    )  # (#frm*Li, d)

                # new
                bs = 32
                # bs = config.batch_size_test.video
                itm_embeds = []

                if config.deep_fusion:
                    if len(topk_idx) % bs != 0:
                        left = len(topk_idx) % bs
                        left_encoder_output = [feat.repeat(left, 1, 1) for feat in encoder_output]
                        left_encoder_att = [
                            torch.ones(feat.size()[:-1], dtype=torch.long).to(
                                device, non_blocking=True
                            )
                            for feat in left_encoder_output
                        ]
                    encoder_output = [feat.repeat(bs, 1, 1) for feat in encoder_output]
                    encoder_att = [
                        torch.ones(feat.size()[:-1], dtype=torch.long).to(
                            device, non_blocking=True
                        )
                        for feat in encoder_output
                    ]
                else:
                    if len(topk_idx) % bs != 0:
                        left = len(topk_idx) % bs
                        left_encoder_output = encoder_output.repeat(left, 1, 1)  # (k=128, #frm*Li, d)
                        left_encoder_att = torch.ones(left_encoder_output.size()[:-1], dtype=torch.long).to(
                            device, non_blocking=True
                        )
                    encoder_output = encoder_output.repeat(bs, 1, 1)  # (k=128, #frm*Li, d)
                    encoder_att = torch.ones(encoder_output.size()[:-1], dtype=torch.long).to(
                        device, non_blocking=True
                    )

                for j in range(0, len(topk_idx), bs):
                    if j + bs > len(topk_idx):
                        output = text_encoder(
                            encoder_embeds=text_feats[topk_idx[j:]],
                            attention_mask=text_atts[topk_idx[j:]],
                            encoder_hidden_states=left_encoder_output,
                            encoder_attention_mask=left_encoder_att,
                            return_dict=True,
                            mode="fusion",
                        )
                    else:
                        output = text_encoder(
                            encoder_embeds=text_feats[topk_idx[j : j + bs]],
                            attention_mask=text_atts[topk_idx[j : j + bs]],
                            encoder_hidden_states=encoder_output,
                            encoder_attention_mask=encoder_att,
                            return_dict=True,
                            mode="fusion",
                        )
                    batch_itm_embeds = output.last_hidden_state[:, 0]
                    itm_embeds.append(batch_itm_embeds)
                itm_embeds = torch.cat(itm_embeds, dim=0)
                # end new

                score = match_head(itm_embeds)[:, 1]
                clip_scores.append(score)

            if len(clip_scores) == 1:
                score = clip_scores[0]
            else:
                raise NotImplementedError(f"len(clip_scores) == {len(clip_scores)}")

            i2t_scores_x[start + i, topk_idx] = score.to(i2t_scores_x.dtype)

        # compute text2media #
        num_text = len(data_loader.dataset.text)
        t2i_scores_x = torch.full((num_text, num_medias), -100.0).to(
            device, torch.float, non_blocking=True
        )

        step = num_text // num_tasks + 1
        start = rank * step
        end = min(num_text, start + step)

        iterator = metric_logger.log_every(t2i_scores[start:end], 100, header)
        logger.info(f"t2i_scores.shape {t2i_scores[start:end].shape}")
        # generate score for each clip, and aggregate all clip scores for a video
        n_clip_per_video = (
            media_feats.shape[1] if not config.deep_fusion else media_feats[0].shape[1]
        )
        for i, sims in enumerate(iterator):
            k = min(len(sims), config.evaluation.k_test)
            topk_sim, topk_idx = sims.topk(k=k, dim=0)

            clip_scores = []
            for clip_idx in range(n_clip_per_video):
                # new
                bs = 32
                # bs = config.batch_size_test.video
                itm_embeds = []
                for j in range(0, len(topk_idx), bs):

                    if config.deep_fusion:
                        encoder_output = [
                            feat[topk_idx[j : j + bs].cpu(), clip_idx].to(device, non_blocking=True)
                            if config.evaluation.eval_offload
                            else feat[topk_idx[j : j + bs], clip_idx]
                            for feat in media_feats
                        ]
                        encoder_att = [
                            torch.ones(feat.size()[:-1], dtype=torch.long).to(
                                device, non_blocking=True
                            )
                            for feat in encoder_output
                        ]
                    else:
                        encoder_output = (
                            media_feats[topk_idx[j : j + bs].cpu(), clip_idx].to(
                                device, non_blocking=True
                            )
                            if config.evaluation.eval_offload
                            else media_feats[topk_idx[j : j + bs], clip_idx]
                        )
                        encoder_att = torch.ones(encoder_output.size()[:-1], dtype=torch.long).to(
                            device, non_blocking=True
                        )

                    repeat_n = (
                        encoder_output.shape[0]
                        if not config.deep_fusion
                        else encoder_output[0].shape[0]
                    )
                    output = text_encoder(
                        encoder_embeds=text_feats[start + i].repeat(repeat_n, 1, 1),
                        attention_mask=text_atts[start + i].repeat(repeat_n, 1),
                        encoder_hidden_states=encoder_output,
                        encoder_attention_mask=encoder_att,
                        return_dict=True,
                        mode="fusion",
                    )

                    batch_itm_embeds = output.last_hidden_state[:, 0]
                    itm_embeds.append(batch_itm_embeds)

                itm_embeds = torch.cat(itm_embeds, dim=0)
                # end new

                score = match_head(itm_embeds)[:, 1]
                clip_scores.append(score)

            if len(clip_scores) == 1:
                score = clip_scores[0]
            else:
                raise NotImplementedError(f"len(clip_scores) == {len(clip_scores)}")

            t2i_scores_x[start + i, topk_idx] = score.to(t2i_scores_x.dtype)

        logger.info("Compute over!!!")
        if config.distributed:
            logger.info("Gather across GPUs!!!")
            # gather across GPUs
            dist.barrier()
            logger.info("dist.barrier()!!!")
            dist.all_reduce(i2t_scores_x, op=dist.ReduceOp.SUM)
            logger.info("dist.all_reduce(i2t_scores_x, op=dist.ReduceOp.SUM) over!!!")
            dist.all_reduce(t2i_scores_x, op=dist.ReduceOp.SUM)
            logger.info("dist.all_reduce(t2i_scores_x, op=dist.ReduceOp.SUM) over!!!")
        
        if use_dsl_for_match:
            i2t_scores_dsl = i2t_scores
            i2t_scores_dsl_T = t2i_scores
            i2t_scores = old_i2t_scores
            t2i_scores = old_t2i_scores
        else:
            i2t_scores_dsl = i2t_scores.float() * i2t_scores.float().softmax(dim=0)
            i2t_scores_dsl_T = i2t_scores.float().T * i2t_scores.float().T.softmax(dim=0)
    else:
        i2t_scores_dsl = i2t_scores.float() * i2t_scores.float().softmax(dim=0)
        i2t_scores_dsl_T = i2t_scores.float().T * i2t_scores.float().T.softmax(dim=0)

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    logger.info(f"Evaluation time {total_time_str}")


    if match_head is not None:
        return (
            i2t_scores_x.softmax(dim=1).cpu().float().numpy(),
            t2i_scores_x.softmax(dim=1).cpu().float().numpy(),
            i2t_scores.softmax(dim=1).cpu().float().numpy(),
            i2t_scores.T.softmax(dim=1).cpu().float().numpy(),
            i2t_scores_dsl.softmax(dim=1).cpu().float().numpy(),
            i2t_scores_dsl_T.softmax(dim=1).cpu().float().numpy()
        )
    else:
        return (
            None,
            None,
            i2t_scores.softmax(dim=1).cpu().float().numpy(),
            i2t_scores.T.softmax(dim=1).cpu().float().numpy(),
            i2t_scores_dsl.softmax(dim=1).cpu().float().numpy(),
            i2t_scores_dsl_T.softmax(dim=1).cpu().float().numpy()
        )
    

@torch.no_grad()
def evaluation_simple(model, data_loader, tokenizer, device, config):
    model.eval()

    metric_logger = MetricLogger(delimiter="  ")
    header = "Evaluation:"
    media_type = data_loader.dataset.media_type

    logger.info(f"Start evaluation for media_type={media_type}")
    assert media_type in ['video'], f"Not implement evaluation of {media_type}"

    logger.info("Computing dual encoder features...")
    start_time = time.time()

    # this computes all features in each GPU
    texts = data_loader.dataset.text
    # max_txt_l of eval depends on data_cofig
    max_txt_l = data_loader.dataset.max_txt_l 

    text_feats, text_atts = extract_text_feats(
        texts, max_txt_l, tokenizer, model, device
    )  # (bsz, Lt, d), (bsz, Lt)

      
    if media_type == 'video':
        image_feats, pooled_image_feats = extract_vision_feats(
            data_loader, model, device, config
        )  # (bsz, 1, #frm*Li, d) or (bsz, #frm, Li, d), (bsz, #frm, d)
        logger.info("Finished vision feature extraction")
        logger.info("Computing ITC scores [dot-product]")
        if config.evaluation.eval_offload:
            # image_feats = image_feats.to(device, non_blocking=True) image_feats will cause OOM!!!
            pooled_image_feats = pooled_image_feats.to(device, non_blocking=True)

        i2t_scores, t2i_scores = get_sim(
            model.vision_proj(pooled_image_feats), model.text_proj(text_feats[:, 0])
        )

        num_medias = len(data_loader.dataset.image)
        
        media_feats = image_feats
        if hasattr(model, "itm_head"):
            match_head = model.itm_head
        else:
            match_head = None

    else:
        raise NotImplementedError(media_type)
    
    logger.info("Computing ITC scores [dot-product], done!")
    
    if match_head is not None:
        i2t_scores_x = torch.full((num_medias, len(texts)), -100.0).to(
            device, torch.float, non_blocking=True
        )

        # computes only part of the scores at each GPU, gather at the end
        logger.info("Rerank dual-encoder results with cross-encoder...")
        num_tasks = get_world_size()
        rank = get_rank()
        # only uses the part associated with the raw eval set
        # compute media2text #
        step = num_medias // num_tasks + 1
        start = rank * step
        end = min(num_medias, start + step)

        text_encoder = model.get_text_encoder()
        iterator = metric_logger.log_every(i2t_scores[start:end], 100, header)
        logger.info(f"i2t_scores.shape {i2t_scores[start:end].shape}")

        # generate score for each clip, and aggregate all clip scores for a video
        n_clip_per_video = (
            media_feats.shape[1] if not config.deep_fusion else media_feats[0].shape[1]
        )

        assert not config.deep_fusion and n_clip_per_video == 1, f"Not implemented for config.deep_fusion={config.deep_fusion} n_clip_per_video={n_clip_per_video}"

        logger.info(
            f"n_clip_per_video={n_clip_per_video}, with eval_frame_ensemble={config.evaluation.eval_frame_ensemble}"
        )

        for i, sims in enumerate(iterator):
            k = min(len(sims), config.evaluation.k_test)
            topk_sim, topk_idx = sims.topk(k=k, dim=0)

            clip_scores = []
            for clip_idx in range(n_clip_per_video):
                if config.deep_fusion:
                    encoder_output = [
                        feat[start + i, clip_idx].to(device, non_blocking=True)
                        if config.evaluation.eval_offload
                        else feat[start + i, clip_idx]
                        for feat in media_feats
                    ]

                else:
                    encoder_output = (
                        media_feats[start + i, clip_idx].to(device, non_blocking=True)
                        if config.evaluation.eval_offload
                        else media_feats[start + i, clip_idx]
                    )  # (#frm*Li, d)

                # new
                bs = 32
                # bs = config.batch_size_test.video
                itm_embeds = []

                if config.deep_fusion:
                    if len(topk_idx) % bs != 0:
                        left = len(topk_idx) % bs
                        left_encoder_output = [feat.repeat(left, 1, 1) for feat in encoder_output]
                        left_encoder_att = [
                            torch.ones(feat.size()[:-1], dtype=torch.long).to(
                                device, non_blocking=True
                            )
                            for feat in left_encoder_output
                        ]
                    encoder_output = [feat.repeat(bs, 1, 1) for feat in encoder_output]
                    encoder_att = [
                        torch.ones(feat.size()[:-1], dtype=torch.long).to(
                            device, non_blocking=True
                        )
                        for feat in encoder_output
                    ]
                else:
                    if len(topk_idx) % bs != 0:
                        left = len(topk_idx) % bs
                        left_encoder_output = encoder_output.repeat(left, 1, 1)  # (k=128, #frm*Li, d)
                        left_encoder_att = torch.ones(left_encoder_output.size()[:-1], dtype=torch.long).to(
                            device, non_blocking=True
                        )
                    encoder_output = encoder_output.repeat(bs, 1, 1)  # (k=128, #frm*Li, d)
                    encoder_att = torch.ones(encoder_output.size()[:-1], dtype=torch.long).to(
                        device, non_blocking=True
                    )

                for j in range(0, len(topk_idx), bs):
                    if j + bs > len(topk_idx):
                        output = text_encoder(
                            encoder_embeds=text_feats[topk_idx[j:]],
                            attention_mask=text_atts[topk_idx[j:]],
                            encoder_hidden_states=left_encoder_output,
                            encoder_attention_mask=left_encoder_att,
                            return_dict=True,
                            mode="fusion",
                        )
                    else:
                        output = text_encoder(
                            encoder_embeds=text_feats[topk_idx[j : j + bs]],
                            attention_mask=text_atts[topk_idx[j : j + bs]],
                            encoder_hidden_states=encoder_output,
                            encoder_attention_mask=encoder_att,
                            return_dict=True,
                            mode="fusion",
                        )
                    batch_itm_embeds = output.last_hidden_state[:, 0]
                    itm_embeds.append(batch_itm_embeds)
                itm_embeds = torch.cat(itm_embeds, dim=0)
                # end new

                score = match_head(itm_embeds)[:, 1]
                clip_scores.append(score)

            if len(clip_scores) == 1:
                score = clip_scores[0]
            else:
                raise NotImplementedError(f"len(clip_scores) == {len(clip_scores)}")

            i2t_scores_x[start + i, topk_idx] = score.to(i2t_scores_x.dtype)

        # compute text2media #
        num_text = len(data_loader.dataset.text)
        t2i_scores_x = torch.full((num_text, num_medias), -100.0).to(
            device, torch.float, non_blocking=True
        )

        step = num_text // num_tasks + 1
        start = rank * step
        end = min(num_text, start + step)

        iterator = metric_logger.log_every(t2i_scores[start:end], 100, header)
        logger.info(f"t2i_scores.shape {t2i_scores[start:end].shape}")
        # generate score for each clip, and aggregate all clip scores for a video
        n_clip_per_video = (
            media_feats.shape[1] if not config.deep_fusion else media_feats[0].shape[1]
        )
        for i, sims in enumerate(iterator):
            k = min(len(sims), config.evaluation.k_test)
            topk_sim, topk_idx = sims.topk(k=k, dim=0)

            clip_scores = []
            for clip_idx in range(n_clip_per_video):
                # new
                bs = 32
                # bs = config.batch_size_test.video
                itm_embeds = []
                for j in range(0, len(topk_idx), bs):

                    if config.deep_fusion:
                        encoder_output = [
                            feat[topk_idx[j : j + bs].cpu(), clip_idx].to(device, non_blocking=True)
                            if config.evaluation.eval_offload
                            else feat[topk_idx[j : j + bs], clip_idx]
                            for feat in media_feats
                        ]
                        encoder_att = [
                            torch.ones(feat.size()[:-1], dtype=torch.long).to(
                                device, non_blocking=True
                            )
                            for feat in encoder_output
                        ]
                    else:
                        encoder_output = (
                            media_feats[topk_idx[j : j + bs].cpu(), clip_idx].to(
                                device, non_blocking=True
                            )
                            if config.evaluation.eval_offload
                            else media_feats[topk_idx[j : j + bs], clip_idx]
                        )
                        encoder_att = torch.ones(encoder_output.size()[:-1], dtype=torch.long).to(
                            device, non_blocking=True
                        )

                    repeat_n = (
                        encoder_output.shape[0]
                        if not config.deep_fusion
                        else encoder_output[0].shape[0]
                    )
                    output = text_encoder(
                        encoder_embeds=text_feats[start + i].repeat(repeat_n, 1, 1),
                        attention_mask=text_atts[start + i].repeat(repeat_n, 1),
                        encoder_hidden_states=encoder_output,
                        encoder_attention_mask=encoder_att,
                        return_dict=True,
                        mode="fusion",
                    )

                    batch_itm_embeds = output.last_hidden_state[:, 0]
                    itm_embeds.append(batch_itm_embeds)

                itm_embeds = torch.cat(itm_embeds, dim=0)
                # end new

                score = match_head(itm_embeds)[:, 1]
                clip_scores.append(score)

            if len(clip_scores) == 1:
                score = clip_scores[0]
            else:
                raise NotImplementedError(f"len(clip_scores) == {len(clip_scores)}")

            t2i_scores_x[start + i, topk_idx] = score.to(t2i_scores_x.dtype)

        logger.info("Compute over!!!")
        if config.distributed:
            logger.info("Gather across GPUs!!!")
            # gather across GPUs
            dist.barrier()
            logger.info("dist.barrier()!!!")
            dist.all_reduce(i2t_scores_x, op=dist.ReduceOp.SUM)
            logger.info("dist.all_reduce(i2t_scores_x, op=dist.ReduceOp.SUM) over!!!")
            dist.all_reduce(t2i_scores_x, op=dist.ReduceOp.SUM)
            logger.info("dist.all_reduce(t2i_scores_x, op=dist.ReduceOp.SUM) over!!!")
        
        i2t_scores_dsl = i2t_scores.float() * i2t_scores.float().softmax(dim=0)
        i2t_scores_dsl_T = i2t_scores.float().T * i2t_scores.float().T.softmax(dim=0)
    else:
        i2t_scores_dsl = i2t_scores.float() * i2t_scores.float().softmax(dim=0)
        i2t_scores_dsl_T = i2t_scores.float().T * i2t_scores.float().T.softmax(dim=0)

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    logger.info(f"Evaluation time {total_time_str}")


    if match_head is not None:
        return (
            i2t_scores_x.softmax(dim=1).cpu().float().numpy(),
            t2i_scores_x.softmax(dim=1).cpu().float().numpy(),
            i2t_scores.softmax(dim=1).cpu().float().numpy(),
            i2t_scores.T.softmax(dim=1).cpu().float().numpy(),
            i2t_scores_dsl.softmax(dim=1).cpu().float().numpy(),
            i2t_scores_dsl_T.softmax(dim=1).cpu().float().numpy()
        )
    else:
        return (
            None,
            None,
            i2t_scores.softmax(dim=1).cpu().float().numpy(),
            i2t_scores.T.softmax(dim=1).cpu().float().numpy(),
            i2t_scores_dsl.softmax(dim=1).cpu().float().numpy(),
            i2t_scores_dsl_T.softmax(dim=1).cpu().float().numpy()
        )
    

@torch.no_grad()
def evaluation_qformer(model, data_loader, tokenizer, device, config):
    model.eval()

    metric_logger = MetricLogger(delimiter="  ")
    header = "Evaluation:"
    dtype = torch.half if config.use_half_precision else torch.float
    media_type = data_loader.dataset.media_type
    logger.info(f"Start evaluation_qformer for media_type={media_type}")
    assert media_type == 'video', f"Not implement evaluation of {media_type}"
    logger.info("Computing dual encoder features...")
    start_time = time.time()

    # this computes all features in each GPU
    texts = data_loader.dataset.text
    # max_txt_l of eval depends on data_cofig
    max_txt_l = data_loader.dataset.max_txt_l 

    text_feats, text_atts, text_ids = extract_text_feats(
        texts, max_txt_l, tokenizer, model, device, return_ids=True
    )  # (bsz, Lt, d), (bsz, Lt)

    if media_type == 'video':
        image_feats, pooled_image_feats = extract_vision_feats(
            data_loader, model, device, config
        )  # (bsz, 1, #frm*Li, d) or (bsz, #frm, Li, d), (bsz, #frm, d)
        logger.info("Finished vision feature extraction")
        logger.info("Computing ITC scores [dot-product]")
        if config.evaluation.eval_offload:
            # image_feats = image_feats.to(device, non_blocking=True) image_feats will cause OOM!!!
            pooled_image_feats = pooled_image_feats.to(device, non_blocking=True)

        if hasattr(model, "q_vision_proj"):
            i2t_scores, t2i_scores = get_sim(
                model.q_vision_proj(pooled_image_feats), model.q_text_proj(text_feats[:, 0])
            )
        else:
            i2t_scores, t2i_scores = get_sim(
                model.vision_proj(pooled_image_feats), model.text_proj(text_feats[:, 0])
            )

        num_medias = len(data_loader.dataset.image)
        
        media_feats = image_feats
        if hasattr(model, "itm_head"):
            match_head = model.itm_head
        elif hasattr(model, "q_itm_head"):
            match_head = model.q_itm_head
        else:
            raise NotImplementedError("you must have a match head in qformer!!!")
    
    logger.info("Computing ITC scores [dot-product], done!")

    if match_head is not None:
        i2t_scores_x = torch.full((num_medias, len(texts)), -100.0).to(
            device, torch.float, non_blocking=True
        )

        # computes only part of the scores at each GPU, gather at the end
        logger.info("Rerank dual-encoder results with cross-encoder...")
        num_tasks = get_world_size()
        rank = get_rank()
        # only uses the part associated with the raw eval set
        # compute image2text #
        step = num_medias // num_tasks + 1
        start = rank * step
        end = min(num_medias, start + step)

        iterator = metric_logger.log_every(i2t_scores[start:end], 100, header)
        logger.info(f"i2t_scores.shape {i2t_scores[start:end].shape}")

        # generate score for each clip, and aggregate all clip scores for a video
        n_clip_per_video = (
            image_feats.shape[1] if not config.deep_fusion else image_feats[0].shape[1]
        )

        assert not config.deep_fusion and n_clip_per_video == 1, f"Not implemented for config.deep_fusion={config.deep_fusion} n_clip_per_video={n_clip_per_video}"

        logger.info(
            f"n_clip_per_video={n_clip_per_video}, with eval_frame_ensemble={config.evaluation.eval_frame_ensemble}"
        )
        for i, sims in enumerate(iterator):
            k = min(len(sims), config.evaluation.k_test)
            topk_sim, topk_idx = sims.topk(k=k, dim=0)

            clip_scores = []
            for clip_idx in range(n_clip_per_video):
                if config.deep_fusion:
                    encoder_output = [
                        feat[start + i, clip_idx].to(device, non_blocking=True)
                        if config.evaluation.eval_offload
                        else feat[start + i, clip_idx]
                        for feat in media_feats
                    ]

                else:
                    encoder_output = (
                        image_feats[start + i, clip_idx].to(device, non_blocking=True)
                        if config.evaluation.eval_offload
                        else image_feats[start + i, clip_idx]
                    )  # (#frm*Li, d)

                # new
                bs = 32
                # bs = config.batch_size_test.video
                itm_embeds = []

                if not config.deep_fusion:  # Create fake list
                    encoder_output = [encoder_output]
                encoder_output = [feat.repeat(bs, 1, 1) for feat in encoder_output]
                encoder_att = [
                    torch.ones(feat.size()[:-1], dtype=torch.long).to(device, non_blocking=True) 
                    for feat in encoder_output
                ]
                
                for j in range(0, len(topk_idx), bs):
                    cur_bs = min(bs, len(topk_idx) - j)
                    encoder_output = [feat[:cur_bs] for feat in encoder_output]
                    encoder_att = [att[:cur_bs] for att in encoder_att]

                    batch_encoder_output = encoder_output if config.deep_fusion else encoder_output[0]
                    batch_encoder_att = encoder_att if config.deep_fusion else encoder_att[0]
                    
                    output = model.vtm_embed(
                        text_ids=text_ids[topk_idx[j:j+bs]],
                        text_atts=text_atts[topk_idx[j:j+bs]],
                        vision_embeds=batch_encoder_output,
                        vision_atts=batch_encoder_att,
                    )


                    itm_embeds.append(output)

                itm_embeds = torch.cat(itm_embeds, dim=0)
                
                score = match_head(itm_embeds)[:, 1]
                clip_scores.append(score)

            if len(clip_scores) == 1:
                score = clip_scores[0]
            else:
                raise NotImplementedError(f"len(clip_scores) == {len(clip_scores)}")

            i2t_scores_x[start + i, topk_idx] = score.to(i2t_scores_x.dtype)

        # compute text2image #
        num_text = len(data_loader.dataset.text)
        t2i_scores_x = torch.full((num_text, len(data_loader.dataset.image)), -100.0).to(
            device, torch.float, non_blocking=True
        )

        step = num_text // num_tasks + 1
        start = rank * step
        end = min(num_text, start + step)

        iterator = metric_logger.log_every(t2i_scores[start:end], 100, header)
        logger.info(f"t2i_scores.shape {t2i_scores[start:end].shape}")
        # generate score for each clip, and aggregate all clip scores for a video
        n_clip_per_video = (
            image_feats.shape[1] if not config.deep_fusion else image_feats[0].shape[1]
        )
        k = config.evaluation.k_test
        logger.info(f"Top-{k} matching")
        for i, sims in enumerate(iterator):
            k = min(len(sims), config.evaluation.k_test)
            topk_sim, topk_idx = sims.topk(k=k, dim=0)

            clip_scores = []
            for clip_idx in range(n_clip_per_video):

                # new
                bs = 32
                # bs = config.batch_size_test.video
                itm_embeds = []
                for j in range(0, len(topk_idx), bs):

                    fake_image_feats = [image_feats] if not config.deep_fusion else image_feats

                    encoder_output = [
                        feat[topk_idx[j : j + bs].cpu(), clip_idx].to(device, non_blocking=True) 
                        if config.evaluation.eval_offload
                        else feat[topk_idx[j : j + bs], clip_idx]
                        for feat in fake_image_feats
                    ]
                    encoder_att = [
                        torch.ones(feat.size()[:-1], dtype=torch.long).to(
                            device, non_blocking=True
                        )
                        for feat in encoder_output
                    ]
                    cur_bs = min(bs, len(topk_idx) - j)

                    batch_encoder_output = encoder_output if config.deep_fusion else encoder_output[0]
                    batch_encoder_att = encoder_att if config.deep_fusion else encoder_att[0]


                    output = model.vtm_embed(
                        text_ids=text_ids[start + i].repeat(cur_bs, 1),
                        text_atts=text_atts[start + i].repeat(cur_bs, 1),
                        vision_embeds=batch_encoder_output,
                        vision_atts=batch_encoder_att,
                    )

                    itm_embeds.append(output)


                itm_embeds = torch.cat(itm_embeds, dim=0)
                # end new

                score = match_head(itm_embeds)[:, 1]
                clip_scores.append(score)

            if len(clip_scores) == 1:
                score = clip_scores[0]
            else:
                raise NotImplementedError(f"len(clip_scores) == {len(clip_scores)}")

            t2i_scores_x[start + i, topk_idx] = score.to(t2i_scores_x.dtype)

        logger.info("Compute over!!!")
        if config.distributed:
            logger.info("Gather across GPUs!!!")
            # gather across GPUs
            dist.barrier()
            logger.info("dist.barrier()!!!")
            dist.all_reduce(i2t_scores_x, op=dist.ReduceOp.SUM)
            logger.info("dist.all_reduce(i2t_scores_x, op=dist.ReduceOp.SUM) over!!!")
            dist.all_reduce(t2i_scores_x, op=dist.ReduceOp.SUM)
            logger.info("dist.all_reduce(t2i_scores_x, op=dist.ReduceOp.SUM) over!!!")

        i2t_scores_dsl = i2t_scores.float() * i2t_scores.float().softmax(dim=0)
        i2t_scores_dsl_T = i2t_scores.float().T * i2t_scores.float().T.softmax(dim=0)

    else:
        i2t_scores_dsl = i2t_scores.float() * i2t_scores.float().softmax(dim=0)
        i2t_scores_dsl_T = i2t_scores.float().T * i2t_scores.float().T.softmax(dim=0)
    
    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    logger.info(f"Evaluation time {total_time_str}")

    i2t_scores_dsl = i2t_scores * i2t_scores.softmax(dim=0)
    i2t_scores_dsl_T = i2t_scores.T * i2t_scores.T.softmax(dim=0)


    if match_head is not None:
        return (
            i2t_scores_x.softmax(dim=1).cpu().float().numpy(),
            t2i_scores_x.softmax(dim=1).cpu().float().numpy(),
            i2t_scores.softmax(dim=1).cpu().float().numpy(),
            i2t_scores.T.softmax(dim=1).cpu().float().numpy(),
            i2t_scores_dsl.softmax(dim=1).cpu().float().numpy(),
            i2t_scores_dsl_T.softmax(dim=1).cpu().float().numpy()
        )
    else:
        return (
            None,
            None,
            i2t_scores.softmax(dim=1).cpu().float().numpy(),
            i2t_scores.T.softmax(dim=1).cpu().float().numpy(),
            i2t_scores_dsl.softmax(dim=1).cpu().float().numpy(),
            i2t_scores_dsl_T.softmax(dim=1).cpu().float().numpy()
        )


@torch.no_grad()
def evaluation_clip(model, data_loader, tokenizer, device, config):
    model.eval()

    metric_logger = MetricLogger(delimiter="  ")
    header = "Evaluation:"
    dtype = torch.half if config.use_half_precision else torch.float
    media_type = data_loader.dataset.media_type
    logger.info(f"Start evaluation_clip for media_type={media_type}")

    logger.info("Computing dual encoder features...")

    # this computes all features in each GPU
    texts = data_loader.dataset.text
    num_text = len(texts)
    text_bs = 256
    text_feats = []
    for i in range(0, num_text, text_bs):
        text = texts[i : min(num_text, i + text_bs)]
        if "internvideo2" in config.model.model_cls.lower():
            text_feat = model.encode_text(tokenizer(text).to(device))
        else:
            raise NotImplementedError
            text_feat = model.encode_text(text)
        text_feats.append(text_feat.cpu())
    text_feats = torch.cat(text_feats, dim=0)
    logger.info("Finished computing text features")

    media_feats = []
    metric_logger = MetricLogger(delimiter="  ")
    header = f"extracting {media_type} feats!!!"
    iterator = metric_logger.log_every(data_loader, 100, header)
    for media, _ in iterator:
        if media_type in ['image', 'video']:
            media = media.to(device, non_blocking=True)
            media_feat = model.encode_vision(media, test=True)
        elif media_type == 'audio':
            media = media.to(device, non_blocking=True)
            media_feat = model.encode_audio(media, test=True)
        elif media_type == 'audio_video':
            raise NotImplementedError(f"Not implement media_type:{media_type}")
        else:
            raise NotImplementedError(f"Not implement media_type:{media_type}")
        
        media_feats.append(media_feat.cpu())

    media_feats = torch.cat(media_feats, dim=0)
    logger.info("Finished feature extraction")
    logger.info("Computing ITC scores [dot-product]")
    # print(media_feats.dtype, text_feats.dtype)
    # print(media_feats.device, text_feats.device)
    i2t_scores, t2i_scores = get_sim(media_feats.float(), text_feats.float())
    del media_feats, text_feats
    logger.info("Computing ITC scores [dot-product], done!")

    i2t_scores_dsl = i2t_scores * i2t_scores.softmax(dim=0)
    i2t_scores_dsl_T = i2t_scores.T * i2t_scores.T.softmax(dim=0)

    return (
        i2t_scores.cpu().float().numpy(),
        i2t_scores.T.cpu().float().numpy(),
        i2t_scores_dsl.cpu().float().numpy(),
        i2t_scores_dsl_T.cpu().float().numpy(),
    )


@torch.no_grad()
def itm_eval(scores_i2t, scores_t2i, txt2img, img2txt):
    # Images->Text
    ranks = np.zeros(scores_i2t.shape[0])
    for index, score in enumerate(scores_i2t):
        inds = np.argsort(score)[::-1]
        # Score
        gt_txt_ids = img2txt[index]
        if isinstance(gt_txt_ids, int):
            ranks[index] = np.where(inds == gt_txt_ids)[0][0]
        else:
            rank = 1e20
            for i in gt_txt_ids:
                tmp = np.where(inds == i)[0][0]
                if tmp < rank:
                    rank = tmp
            ranks[index] = rank

    # Compute metrics
    tr1 = 100.0 * len(np.where(ranks < 1)[0]) / len(ranks)
    tr5 = 100.0 * len(np.where(ranks < 5)[0]) / len(ranks)
    tr10 = 100.0 * len(np.where(ranks < 10)[0]) / len(ranks)

    # Text->Images
    ranks = np.zeros(scores_t2i.shape[0])

    for index, score in enumerate(scores_t2i):
        inds = np.argsort(score)[::-1]
        gt_img_ids = txt2img[index]
        if isinstance(gt_img_ids, int):
            ranks[index] = np.where(inds == gt_img_ids)[0][0]
        else:  # list, used in the case each caption has multiple GT images
            # Score
            rank = 1e20
            for i in gt_img_ids:
                tmp = np.where(inds == i)[0][0]
                if tmp < rank:
                    rank = tmp
            ranks[index] = rank

    # Compute metrics
    ir1 = 100.0 * len(np.where(ranks < 1)[0]) / len(ranks)
    ir5 = 100.0 * len(np.where(ranks < 5)[0]) / len(ranks)
    ir10 = 100.0 * len(np.where(ranks < 10)[0]) / len(ranks)

    tr_mean = (tr1 + tr5 + tr10) / 3
    ir_mean = (ir1 + ir5 + ir10) / 3
    r_mean = (tr_mean + ir_mean) / 2

    eval_result = {
        "v2t_r1": tr1,
        "v2t_r5": tr5,
        "v2t_r10": tr10,
        "v2t_r_mean": tr_mean,
        "t2v_r1": ir1,
        "t2v_r5": ir5,
        "t2v_r10": ir10,
        "t2v_r_mean": ir_mean,
        "r_mean": r_mean,
    }
    eval_result = {k: round(v, 2) for k, v in eval_result.items()}
    return eval_result


