import logging
from os.path import join

import torch
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.nn.functional as F
from einops import rearrange

from dataset import create_dataset, create_loader
from models.utils import tile
from models.vindlu import VindLU
from models.vindlu_vit import VindLU_VIT
from tasks.shared_utils import setup_model
from utils.basic_utils import (MetricLogger, flat_list_of_lists, save_json,
                               setup_seed)
from utils.config_utils import setup_main
from utils.distributed import get_rank

logger = logging.getLogger(__name__)


def get_sim_for_each_question(model, pooled_image_feat, pooled_text_feat):
    """TODO: Docstring for get_sim_for_each_question.

    Args:
        model (TODO): TODO
        pooled_image_feat (torch.Tensor): Shape: [b,t, c]
        pooled_text_feat (torch.Tensor): Shape: [b, n, c]. n is the number of answer candidates.

    Returns: TODO

    """
    image_proj = model.vision_proj
    text_proj = model.text_proj

    image_feat = F.normalize(image_proj(pooled_image_feat), dim=-1)
    text_feat = F.normalize(text_proj(pooled_text_feat), dim=-1)
    sim = torch.matmul(image_feat, rearrange(text_feat, "b n c -> b c n"))  # [b, t, n]
    sim = sim.mean(1) / model.temp  # [b,n]
    sim = F.softmax(sim, dim=1)  # [b, n]
    return sim


def main(config):
    logger.info(f"config: \n{config}")
    logger.info(f"train_file: {config.train_file}")

    setup_seed(config.seed + get_rank())
    device = torch.device(config.device)
    cudnn.benchmark = True

    # create dataloader
    test_dataset = create_dataset("mc_test", config)
    test_loader = create_loader(
        [test_dataset],
        [None],
        batch_size=[config.batch_size_test.video],
        num_workers=[config.num_workers],
        is_trains=[False],
        collate_fns=[None],
    )[0]

    config.scheduler.num_training_steps = 10
    config.scheduler.num_warmup_steps = 10
    model_cls = eval(config.model.get('model_cls', 'VindLU'))
    (
        model,
        model_without_ddp,
        optimizer,
        scheduler,
        scaler,
        tokenizer,
        start_epoch,
        global_step,
    ) = setup_model(
        config,
        model_cls=model_cls,
        has_decoder=False,
        pretrain=False,
        # find_unused_parameters=True,
        find_unused_parameters=False,
    )
    model = model_without_ddp

    logger.info("Start " + "evaluation" if config.evaluate else "training")
    metric_logger = MetricLogger(delimiter="  ")
    iterator = metric_logger.log_every(test_loader, 5, "Evaluation: ")
    num_options_per_q = 5
    all_gt_answers = []
    all_pred_answers = []
    with torch.cuda.amp.autocast(enabled=config.fp16, dtype=torch.bfloat16), torch.no_grad():
        for image, text, ans, ann in iterator:
            image = image.to(device, non_blocking=True)  # bsz
            all_gt_answers.append(ans)
            text = flat_list_of_lists(list(zip(*text)))  # List(str), len=bsz*5
            text_input = tokenizer(
                text,
                padding="max_length",
                truncation=True,
                max_length=config.max_txt_l,
                return_tensors="pt",
            ).to(
                device
            )  # bsz, 5, ?

            # encode text
            text_feat = model.encode_text(text_input)[0]
            # encode image
            image_feat, pooled_image_feat = model.encode_image(image)
            image_feat = tile(image_feat, 0, num_options_per_q)
            image_mask = torch.ones(image_feat.size()[:-1], dtype=torch.long).to(
                device, non_blocking=True
            )
            # pooled_image_feat = tile(pooled_image_feat, 0, num_options_per_q)
            # cross-modal encode
            output = model.get_text_encoder()(
                encoder_embeds=text_feat,
                attention_mask=text_input.attention_mask,
                encoder_hidden_states=image_feat,
                encoder_attention_mask=image_mask,
                return_dict=True,
                mode="fusion",
            )
            itm_embeds = output.last_hidden_state[:, 0]  # [CLS]

            score = model.itm_head(itm_embeds)[:, 1]
            pred_ans = score.view(-1, num_options_per_q).max(1)[1].cpu()
            all_pred_answers.append(pred_ans)

    all_gt_answers = torch.cat(all_gt_answers, 0)
    all_pred_answers = torch.cat(all_pred_answers, 0)
    acc = all_gt_answers == all_pred_answers
    acc = float(torch.sum(acc) / len(acc))
    eval_res = {"test": round(100 * acc, 2)}
    logger.info(f"\n{eval_res}")
    save_json(eval_res, join(config.output_dir, "eval_res.json"))

    dist.barrier()


def main_with_ensemble(config):
    logger.info(f"train_file: {config.train_file}")

    setup_seed(config.seed + get_rank())
    device = torch.device(config.device)
    cudnn.benchmark = True

    # create dataloader
    test_dataset = create_dataset("mc_test", config)
    test_loader = create_loader(
        [test_dataset],
        [None],
        batch_size=[config.inputs.batch_size_test.video],
        num_workers=[config.num_workers],
        is_trains=[False],
        collate_fns=[None],
    )[0]

    config.scheduler.num_training_steps = 10
    config.scheduler.num_warmup_steps = 10
    model_cls = eval(config.model.get('model_cls', 'VindLU'))
    (
        model,
        model_without_ddp,
        optimizer,
        scheduler,
        scaler,
        tokenizer,
        start_epoch,
        global_step,
    ) = setup_model(
        config,
        model_cls=model_cls,
        has_decoder=False,
        pretrain=False,
        # find_unused_parameters=True,
        find_unused_parameters=False,
    )
    model = model_without_ddp

    logger.info("Start " + "evaluation" if config.evaluate else "training")
    metric_logger = MetricLogger(delimiter="  ")
    iterator = metric_logger.log_every(test_loader, 5, "Evaluation: ")
    num_options_per_q = 5
    all_gt_answers = []
    all_pred_answers = []
    predictions = []
    with torch.cuda.amp.autocast(enabled=config.fp16, dtype=torch.bfloat16), torch.no_grad():
        for image, text, ans, ann in iterator:
            image = image.to(device, non_blocking=True)  # bsz
            all_gt_answers.append(ans)
            text = flat_list_of_lists(list(zip(*text)))  # List(str), len=bsz*5
            text_input = tokenizer(
                text,
                padding="max_length",
                truncation=True,
                max_length=config.max_txt_l,
                return_tensors="pt",
            ).to(
                device
            )  # bsz*5, ?

            # encode text
            # [b*5, l, c], [b*5, c]
            text_feat, pooled_text_feat = model.encode_text(text_input)
            # encode image
            if config.evaluation.eval_frame_ensemble == "concat":  # default
                image_feats, pooled_image_feat = model.encode_vision(image, test=True)
                if len(image_feats.shape) == 4:
                    image_feats = rearrange(image_feats, "b t l c -> b (t l) c")
                # (bsz, #frm*L, d), (bsz, #frm, d)
                image_feats = image_feats.unsqueeze(1)  # (bsz, 1, #frm*L, d)
                pooled_image_feat = pooled_image_feat.unsqueeze(1)  # (bsz, 1, #frm, d)
            else:
                assert config.video_input.num_frames == 1, "only support single-frame"
                assert config.evaluation.eval_frame_ensemble in ["mean", "max", "lse"]
                image_feats, pooled_image_feat = model.encode_vision(
                    image
                )  # (bsz, #frm, L, d), (bsz, #frm, d)
            # generate score for each clip, and aggregate all clip scores for a video
            n_clip_per_video = image_feats.shape[1]
            clip_scores = []
            for clip_idx in range(n_clip_per_video):
                image_feat = image_feats[:, clip_idx]
                pooled_image_feat = pooled_image_feat[:, clip_idx]
                image_feat = tile(image_feat, 0, num_options_per_q)
                image_mask = torch.ones(image_feat.size()[:-1], dtype=torch.long).to(
                    device, non_blocking=True
                )

                # contrastive score
                pooled_text_feat = rearrange(
                    pooled_text_feat, "(b n) c -> b n c", n=num_options_per_q
                )
                sim = get_sim_for_each_question(
                    model, pooled_image_feat, pooled_text_feat
                )  # [b, n]
                sim = sim.flatten()  # [b*n,]

                # cross-modal encode
                output = model.get_text_encoder()(
                    encoder_embeds=text_feat,
                    attention_mask=text_input.attention_mask,
                    encoder_hidden_states=image_feat,
                    encoder_attention_mask=image_mask,
                    return_dict=True,
                    mode="fusion",
                )
                itm_embeds = output.last_hidden_state[:, 0]  # [CLS]

                score = F.softmax(model.itm_head(itm_embeds), dim=1)[:, 1]  # [bs*5]
                score = score * 0.7 + sim * 0.3

                clip_scores.append(score)

            if len(clip_scores) == 1:
                score = clip_scores[0]
            else:
                assert config.evaluation.eval_frame_ensemble in ["mean", "max", "lse"]
                clip_scores = torch.stack(clip_scores)  # (#clips, k)
                if config.evaluation.eval_frame_ensemble == "mean":
                    score = clip_scores.mean(0)
                elif config.evaluation.eval_frame_ensemble == "max":
                    score = clip_scores.max(0)[0]
                elif config.evaluation.eval_frame_ensemble == "lse":  # LogSumExp
                    score = torch.logsumexp(clip_scores, dim=0)
                else:
                    raise ValueError(
                        "config.evaluation.eval_frame_ensemble must in [mean, max, lse] when #clip > 1."
                    )

            pred_ans = score.view(-1, num_options_per_q).max(1)[1].cpu()
            all_pred_answers.append(pred_ans)

            # assemble predictions
            ensemble_scores = score.view(-1, num_options_per_q).cpu()  # (bsz, 5)
            if n_clip_per_video > 1:
                clip_scores = clip_scores.view(
                    n_clip_per_video, -1, num_options_per_q
                ).cpu()  # (#clips, bsz, 5)
            for q_idx in range(len(ensemble_scores)):  # bsz
                _pred = dict(
                    video=ann["video"][q_idx],
                    options=[e[q_idx] for e in ann["caption"]],
                    answer=ann["answer"][q_idx].item(),
                    pred_ans_ensemble=pred_ans[q_idx].item(),
                    pred_scores_ensemble=ensemble_scores[q_idx].numpy(),  # (5, )
                )
                # clip scores
                if n_clip_per_video > 1:
                    _pred["pred_scores_frame"] = clip_scores[:, q_idx].numpy()  # (#clips, 5)
                    _pred["pred_ans_frame"] = (
                        clip_scores[:, q_idx].max(1)[1].numpy()
                    )  # (#clips, )
                predictions.append(_pred)

    all_gt_answers = torch.cat(all_gt_answers, 0)
    all_pred_answers = torch.cat(all_pred_answers, 0)
    acc = all_gt_answers == all_pred_answers
    acc = float(torch.sum(acc) / len(acc))
    eval_res = {"test": round(100 * acc, 2)}
    logger.info(f"\n{eval_res}")
    save_json(eval_res, join(config.output_dir, "eval_res.json"))
    torch.save(predictions, join(config.output_dir, "prediction_scores.pth"))

    dist.barrier()


if __name__ == "__main__":
    cfg = setup_main()
    main_with_ensemble(cfg)
