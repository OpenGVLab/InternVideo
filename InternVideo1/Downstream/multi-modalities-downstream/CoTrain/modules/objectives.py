import random
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import glob
import json
import tqdm
import functools
import itertools
from torch.utils.data.distributed import DistributedSampler
import torch.distributed.nn as distnn
from einops import rearrange, repeat

from CoTrain.modules.dist_utils import all_gather
from CoTrain.modules.retrieval_metrics import t2v_metrics, v2t_metrics


def cost_matrix_cosine(x, y, eps=1e-5):
    """Compute cosine distnace across every pairs of x, y (batched)
    [B, L_x, D] [B, L_y, D] -> [B, Lx, Ly]"""
    assert x.dim() == y.dim()
    assert x.size(0) == y.size(0)
    assert x.size(2) == y.size(2)
    x_norm = F.normalize(x, p=2, dim=-1, eps=eps)
    y_norm = F.normalize(y, p=2, dim=-1, eps=eps)
    cosine_sim = x_norm.matmul(y_norm.transpose(1, 2))
    cosine_dist = 1 - cosine_sim
    return cosine_dist


def trace(x):
    """ compute trace of input tensor (batched) """
    b, m, n = x.size()
    assert m == n
    mask = torch.eye(n, dtype=torch.bool, device=x.device).unsqueeze(0).expand_as(x)
    trace = x.masked_select(mask).contiguous().view(b, n).sum(dim=-1, keepdim=False)
    return trace


@torch.no_grad()
def ipot(C, x_len, x_pad, y_len, y_pad, joint_pad, beta, iteration, k):
    """ [B, M, N], [B], [B, M], [B], [B, N], [B, M, N]"""
    b, m, n = C.size()
    sigma = torch.ones(b, m, dtype=C.dtype, device=C.device) / x_len.unsqueeze(1)
    T = torch.ones(b, n, m, dtype=C.dtype, device=C.device)
    A = torch.exp(-C.transpose(1, 2) / beta)

    # mask padded positions
    sigma.masked_fill_(x_pad, 0)
    joint_pad = joint_pad.transpose(1, 2)
    T.masked_fill_(joint_pad, 0)
    A.masked_fill_(joint_pad, 0)

    # broadcastable lengths
    x_len = x_len.unsqueeze(1).unsqueeze(2)
    y_len = y_len.unsqueeze(1).unsqueeze(2)

    # mask to zero out padding in delta and sigma
    x_mask = (x_pad.to(C.dtype) * 1e4).unsqueeze(1)
    y_mask = (y_pad.to(C.dtype) * 1e4).unsqueeze(1)

    for _ in range(iteration):
        Q = A * T  # bs * n * m
        sigma = sigma.view(b, m, 1)
        for _ in range(k):
            delta = 1 / (y_len * Q.matmul(sigma).view(b, 1, n) + y_mask)
            sigma = 1 / (x_len * delta.matmul(Q) + x_mask)
        T = delta.view(b, n, 1) * Q * sigma
    T.masked_fill_(joint_pad, 0)
    return T


def optimal_transport_dist(
    txt_emb, img_emb, txt_pad, img_pad, beta=0.5, iteration=50, k=1
):
    """ [B, M, D], [B, N, D], [B, M], [B, N]"""
    cost = cost_matrix_cosine(txt_emb, img_emb)
    # mask the padded inputs
    joint_pad = txt_pad.unsqueeze(-1) | img_pad.unsqueeze(-2)
    cost.masked_fill_(joint_pad, 0)

    txt_len = (txt_pad.size(1) - txt_pad.sum(dim=1, keepdim=False)).to(dtype=cost.dtype)
    img_len = (img_pad.size(1) - img_pad.sum(dim=1, keepdim=False)).to(dtype=cost.dtype)

    T = ipot(
        cost.detach(), txt_len, txt_pad, img_len, img_pad, joint_pad, beta, iteration, k
    )
    distance = trace(cost.matmul(T.detach()))
    return distance


def compute_mlm(pl_module, batch, infer=None, mode="video"):
    if infer is None:
        infer = pl_module.infer(batch, mask_text=True, mode=mode)
    mlm_logits = pl_module.mlm_score(infer["text_feats"])
    mlm_labels = infer["text_labels"]

    mlm_loss = F.cross_entropy(
        mlm_logits.view(-1, pl_module.hparams.config["vocab_size"]),
        mlm_labels.view(-1),
        ignore_index=-100,
    )

    ret = {
        "mlm_loss": mlm_loss,
        "mlm_logits": mlm_logits,
        "mlm_labels": mlm_labels,
        "mlm_ids": infer["text_ids"],
    }

    phase = "train" if pl_module.training else "val"
    loss = getattr(pl_module, f"{phase}_mlm_loss")(ret["mlm_loss"])
    acc = getattr(pl_module, f"{phase}_mlm_accuracy")(
        ret["mlm_logits"].view(-1, pl_module.hparams.config["vocab_size"]), 
        ret["mlm_labels"].view(-1)
    )
    pl_module.log(f"mlm/{phase}/loss", loss)
    pl_module.log(f"mlm/{phase}/accuracy", acc)

    return ret

def sim_matrix(a, b, eps=1e-8):
    """
    added eps for numerical stability
    """
    a_n, b_n = a.norm(dim=1)[:, None], b.norm(dim=1)[:, None]
    a_norm = a / torch.max(a_n, eps * torch.ones_like(a_n))
    b_norm = b / torch.max(b_n, eps * torch.ones_like(b_n))
    sim_mt = torch.mm(a_norm, b_norm.transpose(0, 1))
    return sim_mt

# == end


# add independent contrastive loss for retrieval

def compute_vtc(pl_module, batch, mode="video"):
    infer_text = pl_module.infer(batch, mask_text=False, mask_video=False, input_text_only=True, mode=mode)
    with torch.cuda.amp.autocast(enabled=False):
        txt_emb = infer_text["text_feats"]
    infer_vision = pl_module.infer(batch, mask_text=False, mask_video=False, input_video_only=True, mode=mode)
    with torch.cuda.amp.autocast(enabled=False):
        img_emb = infer_vision["video_feats"]
    # print(txt_emb.size(), img_emb.size())
    x = sim_matrix(txt_emb[:, 0], img_emb[:, 0])
    temperature = 0.05
    "Assumes input x is similarity matrix of N x M \in [-1, 1], computed using the cosine similarity between normalised vectors"
    i_logsm = F.log_softmax(x / temperature, dim=1)
    j_logsm = F.log_softmax(x.t() / temperature, dim=1)

    # sum over positives
    idiag = torch.diag(i_logsm)
    loss_i = idiag.sum() / len(idiag)

    jdiag = torch.diag(j_logsm)
    loss_j = jdiag.sum() / len(jdiag)

    itc_loss =  - loss_i - loss_j

    ret = {
        "vtc_loss": itc_loss,
    }

    phase = "train" if pl_module.training else "val"
    loss = getattr(pl_module, f"{phase}_vtc_loss")(ret["vtc_loss"])
    pl_module.log(f"vtc/{phase}/loss", loss)

    return ret

# == end

def compute_vtm_wpa(pl_module, batch, mode="video"):
    pos_len = len(batch["text"]) // 2
    neg_len = len(batch["text"]) - pos_len
    vtm_labels = torch.cat([torch.ones(pos_len), torch.zeros(neg_len)]).to(
        pl_module.device
    )
    vtm_labels = vtm_labels[torch.randperm(vtm_labels.size(0))]

    # print(batch.keys())

    vtm_videos = [
        torch.stack(
            [
                ti if vtm_labels[i] == 1 else fi
                for i, (ti, fi) in enumerate(zip(bti, bfi))
            ]
        )
        for bti, bfi in zip(batch["video"], batch["false_video_0"])
    ]

    batch = {k: v for k, v in batch.items()}
    batch["video"] = vtm_videos

    infer = pl_module.infer(batch, mask_text=False, mask_video=False, mode=mode)

    with torch.cuda.amp.autocast(enabled=False):
        txt_emb, img_emb = infer["text_feats"], infer["video_feats"]
        txt_mask, img_mask = infer["text_masks"].bool(), infer["video_masks"].bool()
        for i, _len in enumerate(txt_mask.sum(dim=1)):
            txt_mask[i, _len - 1] = False
        txt_mask[:, 0] = False
        img_mask[:, 0] = False
        if "deit" in pl_module.hparams.config["vit"]:
            img_mask[:, 1] = False
        txt_pad, img_pad = ~txt_mask, ~img_mask

        cost = cost_matrix_cosine(txt_emb.float(), img_emb.float())
        joint_pad = txt_pad.unsqueeze(-1) | img_pad.unsqueeze(-2)
        cost.masked_fill_(joint_pad, 0)

        txt_len = (txt_pad.size(1) - txt_pad.sum(dim=1, keepdim=False)).to(
            dtype=cost.dtype
        )
        img_len = (img_pad.size(1) - img_pad.sum(dim=1, keepdim=False)).to(
            dtype=cost.dtype
        )
        T = ipot(
            cost.detach(), txt_len, txt_pad, img_len, img_pad, joint_pad, 0.5, 50, 1
        )
        distance = trace(cost.matmul(T.detach()))

    dist_pos = distance.masked_select(vtm_labels == 1)
    dist_neg = distance.masked_select(vtm_labels == 0)
    ot_loss = (dist_pos.sum() - dist_neg.sum()) / (dist_pos.size(0) + dist_neg.size(0))

    vtm_logits = pl_module.vtm_score(infer["cls_feats"])
    vtm_loss = F.cross_entropy(vtm_logits, vtm_labels.long())

    ret = {
        "vtm_loss": vtm_loss,
        "vtm_wpa_loss": 0.1 * ot_loss,
        "vtm_logits": vtm_logits,
        "vtm_labels": vtm_labels,
    }

    phase = "train" if pl_module.training else "val"
    loss = getattr(pl_module, f"{phase}_vtm_loss")(ret["vtm_loss"])
    wpa_loss = getattr(pl_module, f"{phase}_vtm_wpa_loss")(ret["vtm_wpa_loss"])
    acc = getattr(pl_module, f"{phase}_vtm_accuracy")(
        ret["vtm_logits"], ret["vtm_labels"]
    )
    pl_module.log(f"vtm/{phase}/loss", loss)
    pl_module.log(f"vtm/{phase}/wpa_loss", wpa_loss)
    pl_module.log(f"vtm/{phase}/accuracy", acc)

    return ret


def compute_imgcls(pl_module, batch):
    infer = pl_module.infer(batch, mask_text=False, mask_video=False)
    imgcls_logits = pl_module.img_classifier(infer["cls_feats"])
    imgcls_labels = batch["label"]
    imgcls_labels = torch.tensor(imgcls_labels).to(pl_module.device).long()
    imgcls_loss = F.cross_entropy(imgcls_logits, imgcls_labels)

    ret = {
        "imgcls_loss": imgcls_loss,
        "imgcls_logits": imgcls_logits,
        "imgcls_labels": imgcls_labels,
    }

    phase = "train" if pl_module.training else "val"
    loss = getattr(pl_module, f"{phase}_imgcls_loss")(ret["imgcls_loss"])
    acc = getattr(pl_module, f"{phase}_imgcls_accuracy")(
        ret["imgcls_logits"], ret["imgcls_labels"]
    )
    pl_module.log(f"imgcls/{phase}/loss", loss)
    pl_module.log(f"imgcls/{phase}/accuracy", acc)

    return ret


# vcr q -> a
def compute_vcr_q2a(pl_module, batch):
    false_len = pl_module.hparams.config["draw_options_text"] - 1
    vtm_labels = torch.tensor(batch["answer"]).to(pl_module.device).long()
    _bs, _t, _c, _h, _w = batch["video"][0].shape
    # for qa
    text_ids = torch.stack(
        [batch[f"options_text_{i}_ids"] for i in range(false_len)], dim=1
    )
    text_masks = torch.stack(
        [batch[f"options_text_{i}_masks"] for i in range(false_len)], dim=1
    )
    text_labels = torch.stack(
        [batch[f"options_text_{i}_labels"] for i in range(false_len)], dim=1
    )

    # concat first option and other options
    text_ids = torch.cat([batch["text_ids"].unsqueeze(1), text_ids], dim=1)
    text_masks = torch.cat([batch["text_masks"].unsqueeze(1), text_masks], dim=1)
    text_labels = torch.cat([batch["text_labels"].unsqueeze(1), text_labels], dim=1)
    videos = batch["video"][0].unsqueeze(1).expand(_bs, false_len + 1, _t, _c, _h, _w)

    infer = pl_module.infer(
        {
            "video": [rearrange(videos, "bs fs t c h w -> (bs fs) t c h w")],
            "text_ids": rearrange(text_ids, "bs fs tl -> (bs fs) tl"),
            "text_masks": rearrange(text_masks, "bs fs tl -> (bs fs) tl"),
            "text_labels": rearrange(text_labels, "bs fs tl -> (bs fs) tl"),
        }
    )
    score = pl_module.rank_output(infer["cls_feats"])[:, 0]
    score = rearrange(score, "(bs fs) -> bs fs", bs=_bs, fs=false_len + 1)
    qa_loss = F.cross_entropy(score, vtm_labels)
    # for qa->r

    reason_len = pl_module.hparams.config["draw_options_text"]
    qar_labels = torch.tensor(batch["reason_answer"]).to(pl_module.device).long()
    _bs, _t, _c, _h, _w = batch["video"][0].shape
    # for qar
    qar_text_ids = torch.stack(
        [batch[f"qar_text_{i}_ids"] for i in range(reason_len)], dim=1
    )
    qar_text_masks = torch.stack(
        [batch[f"qar_text_{i}_masks"] for i in range(reason_len)], dim=1
    )
    qar_text_labels = torch.stack(
        [batch[f"qar_text_{i}_labels"] for i in range(reason_len)], dim=1
    )

    # concat first option and other options
    videos = batch["video"][0].unsqueeze(1).expand(_bs, reason_len, _t, _c, _h, _w)

    qar_infer = pl_module.infer(
        {
            "video": [rearrange(videos, "bs fs t c h w -> (bs fs) t c h w")],
            "text_ids": rearrange(qar_text_ids, "bs fs tl -> (bs fs) tl"),
            "text_masks": rearrange(qar_text_masks, "bs fs tl -> (bs fs) tl"),
            "text_labels": rearrange(qar_text_labels, "bs fs tl -> (bs fs) tl"),
        }
    )
    qar_score = pl_module.rank_output_2(qar_infer["cls_feats"])[:, 0]
    qar_score = rearrange(qar_score, "(bs fs) -> bs fs", bs=_bs, fs=reason_len)
    qar_loss = F.cross_entropy(qar_score, qar_labels)

    # print(score, vtm_labels)
    phase = "train" if pl_module.training else "val"
    qa_acc = getattr(pl_module, f"{phase}_vcr_q2a_accuracy")(
        score, vtm_labels
    )
    qar_acc = getattr(pl_module, f"{phase}_vcr_qar_accuracy")(
        qar_score, qar_labels
    )

    ret = {
        "vcr_q2a_loss": qa_loss,
        "vcr_qar_loss": qar_loss
    }

    phase = "train" if pl_module.training else "val"
    qa_loss = getattr(pl_module, f"{phase}_vcr_q2a_loss")(ret["vcr_q2a_loss"])
    qar_loss = getattr(pl_module, f"{phase}_vcr_qar_loss")(ret["vcr_qar_loss"])

    pl_module.log(f"vcr_q2a/{phase}/loss", qa_loss)
    pl_module.log(f"vcr_qar/{phase}/loss", qar_loss)
    pl_module.log(f"vcr_q2a/{phase}/accuracy", qa_acc)
    pl_module.log(f"vcr_qar/{phase}/accuracy", qar_acc)
    return ret


# vcr qa -> r
def compute_vcr_qa2r(pl_module, batch):
    false_len = pl_module.hparams.config["draw_false_text"] - 1
    # stack video multiple times
    # print(batch["answer"])
    vtm_labels = torch.tensor(batch["answer"]).to(pl_module.device).long()
    _bs, _t, _c, _h, _w = batch["video"][0].shape
    # print(batch.keys())

    text_ids = torch.stack(
        [batch[f"false_text_{i}_ids"] for i in range(false_len)], dim=1
    )
    text_masks = torch.stack(
        [batch[f"false_text_{i}_masks"] for i in range(false_len)], dim=1
    )
    text_labels = torch.stack(
        [batch[f"false_text_{i}_labels"] for i in range(false_len)], dim=1
    )

    # concat first option and other options
    text_ids = torch.cat([batch["text_ids"].unsqueeze(1), text_ids], dim=1)
    text_masks = torch.cat([batch["text_masks"].unsqueeze(1), text_masks], dim=1)
    text_labels = torch.cat([batch["text_labels"].unsqueeze(1), text_labels], dim=1)
    videos = batch["video"][0].unsqueeze(1).expand(_bs, false_len + 1, _t, _c, _h, _w)

    infer = pl_module.infer(
        {
            "video": [rearrange(videos, "bs fs t c h w -> (bs fs) t c h w")],
            "text_ids": rearrange(text_ids, "bs fs tl -> (bs fs) tl"),
            "text_masks": rearrange(text_masks, "bs fs tl -> (bs fs) tl"),
            "text_labels": rearrange(text_labels, "bs fs tl -> (bs fs) tl"),
        }
    )
    score = pl_module.rank_output(infer["cls_feats"])[:, 0]
    score = rearrange(score, "(bs fs) -> bs fs", bs=_bs, fs=false_len + 1)
    loss = F.cross_entropy(score, vtm_labels)

    # print(score, vtm_labels)

    phase = "train" if pl_module.training else "val"
    acc = getattr(pl_module, f"{phase}_multiple_choice_accuracy")(
        score, vtm_labels
    )

    ret = {
        "multiple_choice_loss": loss,
    }

    phase = "train" if pl_module.training else "val"
    loss = getattr(pl_module, f"{phase}_multiple_choice_loss")(ret["multiple_choice_loss"])

    pl_module.log(f"multiple_choice/{phase}/loss", loss)
    pl_module.log(f"multiple_choice/{phase}/accuracy", acc)
    return ret


# mc_vqa
def compute_mc_vqa_q2a(pl_module, batch):
    false_len = pl_module.hparams.config["draw_options_text"] - 1
    vtm_labels = torch.tensor(batch["answer"]).to(pl_module.device).long()
    _bs, _t, _c, _h, _w = batch["video"][0].shape
    # for qa
    text_ids = torch.stack(
        [batch[f"options_text_{i}_ids"] for i in range(false_len)], dim=1
    )
    text_masks = torch.stack(
        [batch[f"options_text_{i}_masks"] for i in range(false_len)], dim=1
    )
    text_labels = torch.stack(
        [batch[f"options_text_{i}_labels"] for i in range(false_len)], dim=1
    )

    # concat first option and other options
    text_ids = torch.cat([batch["text_ids"].unsqueeze(1), text_ids], dim=1)
    text_masks = torch.cat([batch["text_masks"].unsqueeze(1), text_masks], dim=1)
    text_labels = torch.cat([batch["text_labels"].unsqueeze(1), text_labels], dim=1)
    videos = batch["video"][0].unsqueeze(1).expand(_bs, false_len + 1, _t, _c, _h, _w)

    infer = pl_module.infer(
        {
            "video": [rearrange(videos, "bs fs t c h w -> (bs fs) t c h w")],
            "text_ids": rearrange(text_ids, "bs fs tl -> (bs fs) tl"),
            "text_masks": rearrange(text_masks, "bs fs tl -> (bs fs) tl"),
            "text_labels": rearrange(text_labels, "bs fs tl -> (bs fs) tl"),
        }
    )
    ##  v0: use rank output
    # score = pl_module.rank_output(infer["cls_feats"])[:, 0]
    ## v1: use classification head
    # print(infer["cls_feats"].size()) # 40, 768
    score = pl_module.mc_vqa_classifier(infer["cls_feats"])[:, 0]
    score = rearrange(score, "(bs fs) -> bs fs", bs=_bs, fs=false_len + 1)
    qa_loss = F.cross_entropy(score, vtm_labels)
    # print(score, vtm_labels)
    phase = "train" if pl_module.training else "val"
    qa_acc = getattr(pl_module, f"{phase}_mc_vqa_accuracy")(
        score, vtm_labels
    )
    ret = {
        "mc_vqa_loss": qa_loss,
    }

    phase = "train" if pl_module.training else "val"
    qa_loss = getattr(pl_module, f"{phase}_mc_vqa_loss")(ret["mc_vqa_loss"])
    pl_module.log(f"mc_vqa/{phase}/loss", qa_loss)
    pl_module.log(f"mc_vqa/{phase}/accuracy", qa_acc)
    return ret


# msrvtt multiple choice
def compute_multiple_choice(pl_module, batch):
    false_len = pl_module.hparams.config["draw_false_text"] - 1
    # stack image multiple times
    # print(batch["answer"])
    vtm_labels = torch.tensor(batch["answer"]).to(pl_module.device).long()
    _bs, _t, _c, _h, _w = batch["video"][0].shape
    # print(batch.keys())

    texts = [batch[f"false_text_{i}"] for i in range(false_len)]
    text_ids = torch.stack(
        [batch[f"false_text_{i}_ids"] for i in range(false_len)], dim=1
    )
    text_masks = torch.stack(
        [batch[f"false_text_{i}_masks"] for i in range(false_len)], dim=1
    )
    text_labels = torch.stack(
        [batch[f"false_text_{i}_labels"] for i in range(false_len)], dim=1
    )

    # concat first option and other options
    texts = [batch["text"]] + texts
    text_ids = torch.cat([batch["text_ids"].unsqueeze(1), text_ids], dim=1)
    text_masks = torch.cat([batch["text_masks"].unsqueeze(1), text_masks], dim=1)
    text_labels = torch.cat([batch["text_labels"].unsqueeze(1), text_labels], dim=1)

    if "cotrain" in type(pl_module).__name__.lower():
        videos = batch["video"][0].unsqueeze(1).expand(_bs, false_len + 1, _t, _c, _h, _w)
        infer = pl_module.infer(
            {
                "video": [rearrange(videos, "bs fs t c h w -> (bs fs) t c h w")],
                "text_ids": rearrange(text_ids, "bs fs tl -> (bs fs) tl"),
                "text_masks": rearrange(text_masks, "bs fs tl -> (bs fs) tl"),
                "text_labels": rearrange(text_labels, "bs fs tl -> (bs fs) tl"),
            }
        )
        score = pl_module.rank_output(infer["cls_feats"])[:, 0]
        score = rearrange(score, "(bs fs) -> bs fs", bs=_bs, fs=false_len + 1)
    elif "clip" in type(pl_module).__name__.lower():
        if pl_module.mc_type == "vtc":
            videos = batch["video"][0]
            infer = pl_module.infer(
                {
                    "video": [videos],
                    "text": [x for y in zip(*texts) for x in y]
                }
            )
            video_feats = infer["video_feats"]  #  8 * 512
            text_feats = infer["text_feats"]  # 40 * 512

            video_feats = video_feats / video_feats.norm(dim=1, keepdim=True)
            text_feats = text_feats / text_feats.norm(dim=1, keepdim=True)

            text_feats = rearrange(text_feats, "(bs fs) c -> bs fs c", bs=_bs, fs=false_len + 1)
            score = torch.einsum("bc,bfc->bf", video_feats, text_feats) * pl_module.clip.logit_scale.exp()
        # elif pl_module.mc_type == "vtc":
        #     videos = batch["video"][0]
        #     scores = []
        #     for _ in range(int(round(1 / ((1 - pl_module.hparams.config["mim_prob"]))))):
        #         infer = pl_module.infer(
        #             {
        #                 "video": [videos],
        #                 "text": [x for y in zip(*texts) for x in y]
        #             },
        #             mask_video=True,
        #         )
        #         video_feats = infer["video_feats"]  #  8 * 512
        #         text_feats = infer["text_feats"]  # 40 * 512

        #         video_feats = video_feats / video_feats.norm(dim=1, keepdim=True)
        #         text_feats = text_feats / text_feats.norm(dim=1, keepdim=True)

        #         text_feats = rearrange(text_feats, "(bs fs) c -> bs fs c", bs=_bs, fs=false_len + 1)
        #         score = torch.einsum("bc,bfc->bf", video_feats, text_feats) * pl_module.clip.logit_scale.exp()
        #         scores.append(score)
        #     score = sum([x.softmax(dim=-1) for x in scores]) / len(scores)
        elif pl_module.mc_type == "vtc_cap":
            videos = batch["video"][0]
            videos = repeat(videos, 'b t c h w -> (b fs) t c h w', fs=false_len + 1)
            infer = pl_module.infer(
                {
                    "video": [videos],
                    "text": [x for y in zip(*texts) for x in y]
                },
                caption=True,
            )
            feats = infer["cap_logits"]
            feats = feats[torch.arange(feats.shape[0]), infer["text_ids"].argmax(dim=-1)]
            mc_logits = pl_module.rank_output(
                torch.cat([feats, infer["video_feats"], infer["text_feats"]], dim=1))
            score = mc_logits.reshape(_bs, false_len + 1)
    else:
        raise NotImplementedError("Not implemented for model {}".format(pl_module))
    
    loss = F.cross_entropy(score, vtm_labels)

    # print(score, itm_labels)

    phase = "train" if pl_module.training else "val"
    acc = getattr(pl_module, f"{phase}_multiple_choice_accuracy")(
        score, vtm_labels
    )
    # print(acc)
    ret = {
        "multiple_choice_loss": loss,
    }

    phase = "train" if pl_module.training else "val"
    loss = getattr(pl_module, f"{phase}_multiple_choice_loss")(ret["multiple_choice_loss"])

    pl_module.log(f"multiple_choice/{phase}/loss", loss)
    pl_module.log(f"multiple_choice/{phase}/accuracy", acc)
    return ret


def compute_vqa(pl_module, batch):
    infer = pl_module.infer(batch, mask_text=False, mask_video=False)
    vqa_logits = pl_module.vqa_classifier(infer["cls_feats"])
    vqa_targets = torch.zeros(
        len(vqa_logits), pl_module.hparams.config["vqav2_label_size"]
    ).to(pl_module.device)

    vqa_labels = batch["vqa_labels"]
    vqa_scores = batch["vqa_scores"]

    for i, (_label, _score) in enumerate(zip(vqa_labels, vqa_scores)):
        for l, s in zip(_label, _score):
            vqa_targets[i, l] = s

    vqa_loss = (
        F.binary_cross_entropy_with_logits(vqa_logits, vqa_targets)
        * vqa_targets.shape[1]
    )  # https://github.com/jnhwkim/ban-vqa/blob/master/train.py#L19

    ret = {
        "vqa_loss": vqa_loss,
        "vqa_logits": vqa_logits,
        "vqa_targets": vqa_targets,
        "vqa_labels": vqa_labels,
        "vqa_scores": vqa_scores,
    }

    phase = "train" if pl_module.training else "val"
    loss = getattr(pl_module, f"{phase}_vqa_loss")(ret["vqa_loss"])
    score = getattr(pl_module, f"{phase}_vqa_score")(
        ret["vqa_logits"], ret["vqa_targets"]
    )
    pl_module.log(f"vqa/{phase}/loss", loss)
    pl_module.log(f"vqa/{phase}/score", score)

    return ret


# add by vcop
def compute_vcop(pl_module, batch):
    infer = pl_module.infer(batch, mask_text=False, mask_video=False)
    x = infer["vcop_features"]  # BTLC
    b = x.size(0)
    # # v1: simple concat
    # gt_labels = torch.ones(b)
    # idx = torch.randperm(pl_module.hparams.config["num_frames"])  # get random order
    # classes = list(itertools.permutations(list(range(len(idx.tolist())))))
    # label = classes.index(tuple(idx.tolist()))
    # h = x[0, idx, 0].view(1, -1)
    # gt_labels[0] = label
    # for index in range(1, b):
    #     idx = torch.randperm(pl_module.hparams.config["num_frames"])  # get random order
    #     classes = list(itertools.permutations(list(range(len(idx.tolist())))))
    #     label = classes.index(tuple(idx.tolist()))
    #     gt_labels[index] = label
    #     h = torch.cat((h, x[index, idx, 0].view(1, -1)), dim=0)

    # v2: vcop implementation
    gt_labels = torch.ones(b)
    idx = torch.randperm(pl_module.hparams.config["num_frames"])  # get random order
    classes = list(itertools.permutations(list(range(len(idx.tolist())))))
    label = classes.index(tuple(idx.tolist()))
    h = x[0, idx, 0].unsqueeze(0)
    gt_labels[0] = label
    for index in range(1, b):
        idx = torch.randperm(pl_module.hparams.config["num_frames"])  # get random order
        classes = list(itertools.permutations(list(range(len(idx.tolist())))))
        label = classes.index(tuple(idx.tolist()))
        gt_labels[index] = label
        h = torch.cat((h, x[index, idx, 0].unsqueeze(0)), dim=0)
    vcop_logits = pl_module.vcop_classifier(h)
    vcop_labels = gt_labels.to(pl_module.device).long()
    m = nn.Softmax(dim=1)
    if random.random() < 0.01:
        print(m(vcop_logits)[0], vcop_labels[0])
    # print(vcop_labels)
    vcop_loss = F.cross_entropy(vcop_logits, vcop_labels)
    ret = {
        "vcop_loss": vcop_loss,
        "vcop_logits": vcop_logits,
        "vcop_labels": vcop_labels,
    }
    phase = "train" if pl_module.training else "val"
    loss = getattr(pl_module, f"{phase}_vcop_loss")(ret["vcop_loss"])
    pl_module.log(f"vcop/{phase}/loss", loss)
    acc = getattr(pl_module, f"{phase}_vcop_accuracy")(
        ret["vcop_logits"], ret["vcop_labels"], unfilterd=False  # if remove unknown classes
    )
    # print(acc)
    pl_module.log(f"vcop/{phase}/accuracy", acc)
    return ret


# add for dino
def compute_dino(pl_module, batch):
    infer = pl_module.infer(batch, mask_text=False, mask_video=False)
    x = infer["dino_features"]  # BTLC
    b = x.size(0)
    dino_loss = pl_module.dino_loss
    ret = {
        "dino_loss": dino_loss,
        "dino_logits": dino_logits,
        "dino_labels": dino_labels,
    }
    phase = "train" if pl_module.training else "val"
    loss = getattr(pl_module, f"{phase}_dino_loss")(ret["dino_loss"])
    pl_module.log(f"dino/{phase}/loss", loss)
    acc = getattr(pl_module, f"{phase}_dino_accuracy")(
        ret["dino_logits"], ret["dino_labels"], unfilterd=False  # if remove unknown classes
    )
    pl_module.log(f"dino/{phase}/accuracy", acc)
    return ret

# add by msrvtt qa
def compute_openend_vqa(pl_module, batch):
    phase = "train" if pl_module.training else "val"
    # batch["false_video_0"] = batch["false_video_0"][0]
    if "allinone" in type(pl_module).__name__.lower():
        batch["video"] = batch["video"][0]
        infer = pl_module.infer(batch, mask_text=False, mask_video=False, mode="video")
        vqa_logits = pl_module.vqa_classifier(infer["cls_feats"])
    elif "clip" in type(pl_module).__name__.lower():
        if pl_module.qa_type == "vtc":
            infer = pl_module.infer(batch, mask_text=False, mask_video=False, mode="video")
            vqa_logits = pl_module.vqa_classifier(
                torch.cat([infer["video_feats"], infer["text_feats"]], dim=1)
            )
        elif pl_module.qa_type == "cap":
            infer = pl_module.infer(batch, mask_text=False, mask_video=False, caption=True, mode="video")
            # Take the feats of the eot_token
            feats = feats[torch.arange(feats.shape[0]), infer["text_ids"].argmax(dim=-1)]
            vqa_logits = pl_module.vqa_classifier(feats)
        elif pl_module.qa_type == "vtc_cap":
            infer = pl_module.infer(batch, mask_text=False, mask_video=False, caption=True, mode="video")
            feats = infer["cap_logits"]
            feats = feats[torch.arange(feats.shape[0]), infer["text_ids"].argmax(dim=-1)]
            vqa_logits = pl_module.vqa_classifier(
                torch.cat([feats, infer["video_feats"], infer["text_feats"]], dim=1))
        elif pl_module.qa_type == "vtc_mlm":
            del batch["clip_text_ids"]
            assert "clip_text_ids" not in batch
            batch["text"] = [f"Question: {q} Answer: " for q in batch["text"]]
            infer = pl_module.infer(batch, mask_text=True, mask_video=False, mode="video")
            # vqa_logits = pl_module.mlm_score(infer["text_feats"])

            # vqa_logits = vqa_logits[torch.arange(vqa_logits.shape[0]), infer["text_ids"].argmax(dim=-1)]

            # id_idxes = batch["ans_clip_id"][0]
            # vqa_logits = vqa_logits[:, id_idxes]

            feats = infer["text_feats"]
            feats = feats[torch.arange(feats.shape[0]), infer["text_ids"].argmax(dim=-1)]
            vqa_logits = pl_module.vqa_classifier(
                torch.cat([feats, infer["video_feats"], infer["text_contrastive_feats"]], dim=1)
            )
            # vqa_logits = (vqa_logits + vqa_logits_all[:, :vqa_logits.size(1)]) / 2
        elif pl_module.qa_type in ["zs", "mlm"]:
            del batch["clip_text_ids"]
            assert "clip_text_ids" not in batch
            batch["text"] = [f"Question: {q} Answer: " for q in batch["text"]]
            infer = pl_module.infer(batch, mask_text=True, mask_video=False, mode="video")
            vqa_logits = pl_module.mlm_score(infer["text_feats"])

            vqa_logits = vqa_logits[torch.arange(vqa_logits.shape[0]), infer["text_ids"].argmax(dim=-1)]

            id_idxes = batch["ans_clip_id"][0]
            vqa_logits = vqa_logits[:, id_idxes]
    else:
        raise NotImplementedError("Not implemented for model {}".format(pl_module))
    vqa_labels = torch.tensor(batch["vqa_labels"]).to(pl_module.device).long()
    # print(vqa_logits.size())
    # print(vqa_labels)
    vqa_loss = F.cross_entropy(vqa_logits, vqa_labels)
    ret = {
        "vqa_loss": vqa_loss,
        "vqa_logits": vqa_logits,
        "vqa_labels": vqa_labels,
    }
    loss = getattr(pl_module, f"{phase}_vqa_loss")(ret["vqa_loss"])
    pl_module.log(f"vqa/{phase}/loss", loss)
    acc = getattr(pl_module, f"{phase}_openend_vqa_accuracy")(
        ret["vqa_logits"].clone(), ret["vqa_labels"].clone()  # if remove unknown classes
    )
    pl_module.log(f"vqa/{phase}/accuracy", acc)
    return ret


def compute_nlvr2(pl_module, batch):
    infer1 = pl_module.infer(
        batch, mask_text=False, mask_video=False, video_token_type_idx=1
    )
    infer2 = pl_module.infer(
        batch, mask_text=False, mask_video=False, video_token_type_idx=2
    )

    cls_feats = torch.cat([infer1["cls_feats"], infer2["cls_feats"]], dim=-1)
    nlvr2_logits = pl_module.nlvr2_classifier(cls_feats)

    nlvr2_labels = batch["answers"]
    nlvr2_labels = torch.tensor(nlvr2_labels).to(pl_module.device).long()
    nlvr2_loss = F.cross_entropy(nlvr2_logits, nlvr2_labels)

    ret = {
        "nlvr2_loss": nlvr2_loss,
        "nlvr2_logits": nlvr2_logits,
        "nlvr2_labels": nlvr2_labels,
    }

    phase = "train" if pl_module.training else "val"

    if phase == "train":
        loss = getattr(pl_module, f"{phase}_nlvr2_loss")(ret["nlvr2_loss"])
        acc = getattr(pl_module, f"{phase}_nlvr2_accuracy")(
            ret["nlvr2_logits"], ret["nlvr2_labels"]
        )
        pl_module.log(f"nlvr2/{phase}/loss", loss)
        pl_module.log(f"nlvr2/{phase}/accuracy", acc)
    else:
        dev_batches = [i for i, n in enumerate(batch["table_name"]) if "dev" in n]
        test_batches = [i for i, n in enumerate(batch["table_name"]) if "test" in n]

        if dev_batches:
            dev_loss = getattr(pl_module, f"dev_nlvr2_loss")(
                F.cross_entropy(
                    ret["nlvr2_logits"][dev_batches], ret["nlvr2_labels"][dev_batches]
                )
            )
            dev_acc = getattr(pl_module, f"dev_nlvr2_accuracy")(
                ret["nlvr2_logits"][dev_batches], ret["nlvr2_labels"][dev_batches]
            )
            pl_module.log(f"nlvr2/dev/loss", dev_loss)
            pl_module.log(f"nlvr2/dev/accuracy", dev_acc)
        if test_batches:
            test_loss = getattr(pl_module, f"test_nlvr2_loss")(
                F.cross_entropy(
                    ret["nlvr2_logits"][test_batches], ret["nlvr2_labels"][test_batches]
                )
            )
            test_acc = getattr(pl_module, f"test_nlvr2_accuracy")(
                ret["nlvr2_logits"][test_batches], ret["nlvr2_labels"][test_batches]
            )
            pl_module.log(f"nlvr2/test/loss", test_loss)
            pl_module.log(f"nlvr2/test/accuracy", test_acc)

    return ret


def compute_irtr(pl_module, batch):
    is_training_phase = pl_module.training
    # modify to module
    _bs, _t, _c, _h, _w = batch["video"][0].shape
    false_len = pl_module.hparams.config["draw_false_text"]
    text_ids = torch.stack(
        [batch[f"false_text_{i}_ids"] for i in range(false_len)], dim=1
    )
    text_masks = torch.stack(
        [batch[f"false_text_{i}_masks"] for i in range(false_len)], dim=1
    )
    text_labels = torch.stack(
        [batch[f"false_text_{i}_labels"] for i in range(false_len)], dim=1
    )

    text_ids = torch.cat([batch["text_ids"].unsqueeze(1), text_ids], dim=1)
    text_masks = torch.cat([batch["text_masks"].unsqueeze(1), text_masks], dim=1)
    text_labels = torch.cat([batch["text_labels"].unsqueeze(1), text_labels], dim=1)
    videos = batch["video"][0].unsqueeze(1).expand(_bs, false_len + 1, _t, _c, _h, _w)

    infer = pl_module.infer(
        {
            "video": [rearrange(videos, "bs fs t c h w -> (bs fs) t c h w")],
            "text_ids": rearrange(text_ids, "bs fs tl -> (bs fs) tl"),
            "text_masks": rearrange(text_masks, "bs fs tl -> (bs fs) tl"),
            "text_labels": rearrange(text_labels, "bs fs tl -> (bs fs) tl"),
        }
    )
    score = pl_module.rank_output(infer["cls_feats"])[:, 0]
    score = rearrange(score, "(bs fs) -> bs fs", bs=_bs, fs=false_len + 1)
    answer = torch.zeros(_bs).to(score).long()
    irtr_loss = F.cross_entropy(score, answer)

    ret = {
        "irtr_loss": irtr_loss,
    }

    phase = "train" if pl_module.training else "val"
    irtr_loss = getattr(pl_module, f"{phase}_irtr_loss")(ret["irtr_loss"])

    pl_module.log(f"irtr/{phase}/irtr_loss", irtr_loss)

    return ret


# use this method to achievt multiple view testing
@torch.no_grad()
def compute_irtr_recall(pl_module):
    text_dset = pl_module.trainer.datamodule.dms[0].make_no_false_val_dset()
    text_dset.tokenizer = pl_module.trainer.datamodule.dms[0].tokenizer
    text_loader = torch.utils.data.DataLoader(
        text_dset,
        batch_size=64,
        num_workers=pl_module.hparams.config["num_workers"],
        pin_memory=True,
        collate_fn=functools.partial(
            text_dset.collate,
            mlm_collator=pl_module.trainer.datamodule.dms[0].mlm_collator,
        ),
    )

    video_dset = pl_module.trainer.datamodule.dms[0].make_no_false_val_dset(
        video_only=True
    )
    video_dset.tokenizer = pl_module.trainer.datamodule.dms[0].tokenizer
    dist_sampler = DistributedSampler(video_dset, shuffle=False)
    video_loader = torch.utils.data.DataLoader(
        video_dset,
        batch_size=1,
        num_workers=pl_module.hparams.config["num_workers"],
        sampler=dist_sampler,
        pin_memory=True,
        collate_fn=functools.partial(
            video_dset.collate,
            mlm_collator=pl_module.trainer.datamodule.dms[0].mlm_collator,
        ),
    )

    text_preload = list()
    for _b in tqdm.tqdm(text_loader, desc="text prefetch loop"):
        text_preload.append(
            {
                "text_ids": _b["text_ids"].to(pl_module.device),
                "text_masks": _b["text_masks"].to(pl_module.device),
                "text_labels": _b["text_labels"].to(pl_module.device),
                "img_index": _b["img_index"],
            }
        )

    tiids = list()
    for pre in text_preload:
        tiids += pre["img_index"]
    tiids = torch.tensor(tiids)

    video_preload = list()
    for _b in tqdm.tqdm(video_loader, desc="video prefetch loop"):
        video = _b["video"][0]
        # print(video.size())
        (ie, im, _, _) = pl_module.transformer.visual_embed(
            video.to(pl_module.device),
            max_video_len=pl_module.hparams.config["max_video_len"],
            mask_it=False,
        )
        video_preload.append((ie, im, _b["img_index"][0]))

    rank_scores = list()
    rank_iids = list()

    for img_batch in tqdm.tqdm(video_preload, desc="rank loop"):
        _ie, _im, _iid = img_batch
        num_frames, l, c = _ie.shape

        # print(_ie.size())  # 1x197x168
        # print(_im.size())  # 1x197
        _ie.unsqueeze(0)
        _im.unsqueeze(0)
        img_batch_score = list()
        for txt_batch in text_preload:
            fblen = len(txt_batch["text_ids"])
            ie = _ie.expand(fblen, num_frames, l, c)
            # print(ie.size())
            im = _im.expand(fblen, num_frames, l)
            ie = ie.contiguous().view(-1, l, c)
            im = im.contiguous().view(-1, l)

            with torch.cuda.amp.autocast():
                score = pl_module.rank_output(
                    pl_module.infer(
                        {
                            "text_ids": txt_batch["text_ids"],
                            "text_masks": txt_batch["text_masks"],
                            "text_labels": txt_batch["text_labels"],
                        },
                        video_embeds=ie,
                        v_masks=im,
                    )["cls_feats"]
                )[:, 0]

            img_batch_score.append(score)

        img_batch_score = torch.cat(img_batch_score)
        rank_scores.append(img_batch_score.cpu().tolist())
        rank_iids.append(_iid)

    torch.distributed.barrier()
    gather_rank_scores = all_gather(rank_scores)
    gather_rank_iids = all_gather(rank_iids)

    iids = torch.tensor(gather_rank_iids)
    iids = iids.view(-1)
    scores = torch.tensor(gather_rank_scores)
    scores = scores.view(len(iids), -1)

    topk10 = scores.topk(10, dim=1)
    topk5 = scores.topk(5, dim=1)
    topk1 = scores.topk(1, dim=1)
    topk10_iids = tiids[topk10.indices]
    topk5_iids = tiids[topk5.indices]
    topk1_iids = tiids[topk1.indices]

    tr_r10 = (iids.unsqueeze(1) == topk10_iids).float().max(dim=1)[0].mean()
    tr_r5 = (iids.unsqueeze(1) == topk5_iids).float().max(dim=1)[0].mean()
    tr_r1 = (iids.unsqueeze(1) == topk1_iids).float().max(dim=1)[0].mean()

    topk10 = scores.topk(10, dim=0)
    topk5 = scores.topk(5, dim=0)
    topk1 = scores.topk(1, dim=0)
    topk10_iids = iids[topk10.indices]
    topk5_iids = iids[topk5.indices]
    topk1_iids = iids[topk1.indices]

    ir_r10 = (tiids.unsqueeze(0) == topk10_iids).float().max(dim=0)[0].mean()
    ir_r5 = (tiids.unsqueeze(0) == topk5_iids).float().max(dim=0)[0].mean()
    ir_r1 = (tiids.unsqueeze(0) == topk1_iids).float().max(dim=0)[0].mean()

    return (ir_r1, ir_r5, ir_r10, tr_r1, tr_r5, tr_r10)


@torch.no_grad()
def compute_decouple_irtr_recall(pl_module):
    sample_dset = pl_module.trainer.datamodule.dms[0].make_no_false_val_dset(
    )
    sample_dset.tokenizer = pl_module.trainer.datamodule.dms[0].tokenizer
    dist_sampler = DistributedSampler(sample_dset, shuffle=False)
    sample_loader = torch.utils.data.DataLoader(
        sample_dset,
        batch_size=1,
        num_workers=pl_module.hparams.config["num_workers"],
        sampler=dist_sampler,
        pin_memory=True,
        collate_fn=functools.partial(
            sample_dset.collate,
            mlm_collator=pl_module.trainer.datamodule.dms[0].mlm_collator,
        ),
    )

    text_preload = list()
    text_embed_arr = []
    vid_embed_arr = []
    count = 0
    with torch.no_grad():
        for _b in tqdm.tqdm(sample_loader, desc="text&video prefetch loop"):
            # print(_b)
            # print(_b.keys())
            _b["text_ids"] =  _b["text_ids"].to(pl_module.device)
            _b["text_masks"] =  _b["text_masks"].to(pl_module.device)
            _b["text_labels"] =  _b["text_labels"].to(pl_module.device)
            _b["video"][0] = _b["video"][0].to(pl_module.device)

            infer = pl_module.infer(_b, mask_text=False, mask_video=False)
            with torch.cuda.amp.autocast(enabled=False):
                text_embed, vid_embed = infer["text_retrieval_feats"], infer["video_retrieval_feats"]
                if vid_embed is not None:
                    vid_embed_all = [torch.zeros_like(vid_embed) for _ in range(pl_module.hparams.config["num_gpus"])]
                    torch.distributed.all_gather(vid_embed_all, vid_embed)
                    vid_embed_all = torch.cat(vid_embed_all, dim=0)
                if text_embed is not None:
                    text_embed_all = [torch.zeros_like(text_embed) for _ in range(pl_module.hparams.config["num_gpus"])]
                    torch.distributed.all_gather(text_embed_all, text_embed)
                    text_embed_all = torch.cat(text_embed_all, dim=0)
                text_embed_arr.append(text_embed_all.cpu())
                vid_embed_arr.append(vid_embed_all.cpu())
                count += 1
    text_embeds = torch.cat(text_embed_arr)
    vid_embeds = torch.cat(vid_embed_arr)
    # print(text_embeds.size(), vid_embeds.size())
    st2sv_sims = sim_matrix(text_embeds, vid_embeds).detach().cpu().numpy()
    for metric in [t2v_metrics, v2t_metrics]:
        metric_name = metric.__name__
        metrics = metric(st2sv_sims)
        if metric == t2v_metrics:
            tr_r1, tr_r5, tr_r10, tr_r50 = metrics["R1"], metrics["R5"], metrics["R10"], metrics["R50"]
        else:
            ir_r1, ir_r5, ir_r10, ir_r50 = metrics["R1"], metrics["R5"], metrics["R10"], metrics["R50"]
        # msg += f"MedR: {metrics['MedR']:g}, MeanR: {metrics['MeanR']:.1f}"
    return (ir_r1, ir_r5, ir_r10, tr_r1, tr_r5, tr_r10)


@torch.no_grad()
def compute_zero_shot_classify_recall(pl_module, batch):
    # process all prompt action label into text representations
    false_len = pl_module.hparams.config["draw_false_text"] - 1
    # stack video multiple times
    # print(batch["answer"])
    vtm_labels = torch.tensor(batch["answer"]).to(pl_module.device).long()
    _bs, _t, _c, _h, _w = batch["video"][0].shape
    # print(batch.keys())

    text_ids = torch.stack(
        [batch[f"false_text_{i}_ids"] for i in range(false_len)], dim=1
    )
    text_masks = torch.stack(
        [batch[f"false_text_{i}_masks"] for i in range(false_len)], dim=1
    )
    text_labels = torch.stack(
        [batch[f"false_text_{i}_labels"] for i in range(false_len)], dim=1
    )

    # concat first option and other options
    text_ids = torch.cat([batch["text_ids"].unsqueeze(1), text_ids], dim=1)
    text_masks = torch.cat([batch["text_masks"].unsqueeze(1), text_masks], dim=1)
    text_labels = torch.cat([batch["text_labels"].unsqueeze(1), text_labels], dim=1)
    videos = batch["video"][0].unsqueeze(1).expand(_bs, false_len + 1, _t, _c, _h, _w)

    infer = pl_module.infer(
        {
            "video": [rearrange(videos, "bs fs t c h w -> (bs fs) t c h w")],
            "text_ids": rearrange(text_ids, "bs fs tl -> (bs fs) tl"),
            "text_masks": rearrange(text_masks, "bs fs tl -> (bs fs) tl"),
            "text_labels": rearrange(text_labels, "bs fs tl -> (bs fs) tl"),
        }
    )
    score = pl_module.rank_output(infer["cls_feats"])[:, 0]
    score = rearrange(score, "(bs fs) -> bs fs", bs=_bs, fs=false_len + 1)
    loss = F.cross_entropy(score, vtm_labels)

    # print(score, vtm_labels)

    phase = "train" if pl_module.training else "val"
    acc = getattr(pl_module, f"{phase}_zero_shot_accuracy")(
        score, vtm_labels
    )
    # print(acc)
    ret = {
        "multiple_choice_loss": loss,
    }

    phase = "train" if pl_module.training else "val"
    loss = getattr(pl_module, f"{phase}_multiple_choice_loss")(ret["multiple_choice_loss"])

    pl_module.log(f"multiple_choice/{phase}/loss", loss)
    pl_module.log(f"multiple_choice/{phase}/accuracy", acc)
    return acc


# for ind itc
@torch.no_grad()
def compute_ind_irtr_recall(pl_module):
    num_views = pl_module.hparams.config["retrieval_views"]
    text_embed_arr_multi = []
    vid_embed_arr_multi = []
    for i in range(num_views):
        sample_dset = pl_module.trainer.datamodule.video_dms[0].make_no_false_val_dset(
        )
        sample_dset.tokenizer = pl_module.trainer.datamodule.video_dms[0].tokenizer
        dist_sampler = DistributedSampler(sample_dset, shuffle=False)
        sample_loader = torch.utils.data.DataLoader(
            sample_dset,
            batch_size=1,
            num_workers=pl_module.hparams.config["num_workers"],
            sampler=dist_sampler,
            pin_memory=True,
            collate_fn=functools.partial(
                sample_dset.collate,
                mlm_collator=pl_module.trainer.datamodule.video_dms[0].mlm_collator,
            ),
        )
        text_preload = list()
        text_embed_arr = []
        vid_embed_arr = []
        count = 0
        with torch.no_grad():
            for _b in tqdm.tqdm(sample_loader, desc="text&video prefetch loop"):
                # print(_b)
                # print(_b.keys())
                _b["text_ids"] = _b["text_ids"].to(pl_module.device)
                _b["text_masks"] = _b["text_masks"].to(pl_module.device)
                _b["text_labels"] = _b["text_labels"].to(pl_module.device)
                _b["video"][0] = _b["video"][0].to(pl_module.device)

                # infer = pl_module.infer(_b, mask_text=False, mask_video=False)

                infer_text = pl_module.infer(_b, mask_text=False, mask_video=False, input_text_only=True)
                infer_vision = pl_module.infer(_b, mask_text=False, mask_video=False, input_video_only=True)

                with torch.cuda.amp.autocast(enabled=False):
                    # text_embed, vid_embed = infer_text["raw_cls_feats"], infer_vision["raw_cls_feats"]
                    text_embed, vid_embed = infer_text["text_feats"][:, 0], infer_vision["video_feats"][:, 0]
                    if vid_embed is not None:
                        vid_embed_all = [torch.zeros_like(vid_embed) for _ in range(pl_module.hparams.config["num_gpus"])]
                        torch.distributed.all_gather(vid_embed_all, vid_embed)
                        vid_embed_all = torch.cat(vid_embed_all, dim=0)
                    if text_embed is not None:
                        text_embed_all = [torch.zeros_like(text_embed) for _ in range(pl_module.hparams.config["num_gpus"])]
                        torch.distributed.all_gather(text_embed_all, text_embed)
                        text_embed_all = torch.cat(text_embed_all, dim=0)
                    text_embed_arr.append(text_embed_all.cpu())
                    vid_embed_arr.append(vid_embed_all.cpu())
                    count += 1
        text_embeds = torch.cat(text_embed_arr)
        vid_embeds = torch.cat(vid_embed_arr)
        # append for multi view
        text_embed_arr_multi.append(text_embeds)
        vid_embed_arr_multi.append(vid_embeds)
        # print(text_embeds.size(), vid_embeds.size())
    for j in range(len(text_embed_arr_multi)):
        if j == 0:
            st2sv_sims = sim_matrix(text_embed_arr_multi[j], vid_embed_arr_multi[j]).detach().cpu().numpy() / len(text_embed_arr_multi)
        else:
            st2sv_sims += sim_matrix(text_embed_arr_multi[j], vid_embed_arr_multi[j]).detach().cpu().numpy() / len(text_embed_arr_multi)
    # st2sv_sims = sim_matrix(text_embeds, vid_embeds).detach().cpu().numpy()
    for metric in [t2v_metrics, v2t_metrics]:
        metric_name = metric.__name__
        metrics = metric(st2sv_sims)
        if metric == t2v_metrics:
            tr_r1, tr_r5, tr_r10, tr_r50 = metrics["R1"], metrics["R5"], metrics["R10"], metrics["R50"]
        else:
            ir_r1, ir_r5, ir_r10, ir_r50 = metrics["R1"], metrics["R5"], metrics["R10"], metrics["R50"]
        # msg += f"MedR: {metrics['MedR']:g}, MeanR: {metrics['MeanR']:.1f}"
    return (ir_r1, ir_r5, ir_r10, tr_r1, tr_r5, tr_r10)


def compute_contrastive(pl_module, batch, return_infer=False, mask_text=False, mask_video=False, caption=False, mode="video"):
    infer = pl_module.infer(batch, mask_text=mask_text, mask_video=mask_video, caption=caption, mode=mode)
    if mask_text:
        text_feats = infer["text_contrastive_feats"]
    else:
        text_feats = infer["text_feats"]
    video_feats = infer["video_feats"]

    if text_feats.ndim == 3:  # [B, N, C] -> [B, C]
        text_feats = text_feats.mean(1)
    if video_feats.ndim == 3:
        video_feats= video_feats.mean(1)
    
    # Normalize the feature
    video_feats = video_feats / video_feats.norm(dim=1, keepdim=True)
    text_feats = text_feats / text_feats.norm(dim=1, keepdim=True)
    
    if not pl_module.mmt:
        # Plain contrastive
        # # TODO: Handle logit_scale when model has no logit_scale
        video_feats = distnn.all_gather(video_feats)
        text_feats = distnn.all_gather(text_feats)

        if not isinstance(video_feats, torch.Tensor) or video_feats.ndim == 3:
            video_feats = torch.cat(list(video_feats))
            text_feats = torch.cat(list(text_feats))

        image_logits = video_feats @ text_feats.t() * pl_module.clip.logit_scale.exp()
        text_logits = image_logits.t()

        ground_truth = torch.arange(len(image_logits)).long().to(image_logits.device)
        loss = (F.cross_entropy(image_logits, ground_truth) + F.cross_entropy(text_logits, ground_truth)).div(2)
    else:
        text_feats_k = infer["text_feats_k"]
        video_feats_k = infer["video_feats_k"]
        if video_feats_k.ndim == 3:
            video_feats_k = video_feats_k.mean(1)
        if text_feats_k.ndim == 3:
            text_feats_k = text_feats_k.mean(1)

        video_feats_k = video_feats_k / video_feats_k.norm(dim=1, keepdim=True)
        text_feats_k = text_feats_k / text_feats_k.norm(dim=1, keepdim=True)

        video_l_pos = torch.einsum('nc, nc->n', video_feats, text_feats_k).unsqueeze(-1)
        video_l_neg = torch.einsum('nc, ck->nk', video_feats, pl_module.queue_text)
        image_logits = torch.cat([video_l_pos, video_l_neg], dim=1) * pl_module.clip.logit_scale.exp()

        text_l_pos = torch.einsum('nc, nc->n', text_feats, video_feats_k).unsqueeze(-1)
        text_l_neg = torch.einsum('nc, ck->nk', text_feats, pl_module.queue_visual)
        text_logits = torch.cat([text_l_pos, text_l_neg], dim=1) * pl_module.clip.logit_scale.exp()

        ground_truth = torch.zeros(image_logits.shape[0], dtype=torch.long).to(image_logits.device)
        loss = (F.cross_entropy(image_logits, ground_truth) + F.cross_entropy(text_logits, ground_truth)).div(2)

        pl_module._dequeue_and_enqueue(text_feats_k, video_feats_k)


    ret = {
        "contrastive_loss": loss,
        "contrastive_image_logits": image_logits,
        "contrastive_text_logits": text_logits,
        "contrastive_labels": ground_truth,
    }

    phase = "train" if pl_module.training else "val"
    loss = getattr(pl_module, f"{phase}_contrastive_loss")(ret["contrastive_loss"])
    acc_image = getattr(pl_module, f"{phase}_contrastive_image_accuracy")(
        ret["contrastive_image_logits"], ret["contrastive_labels"]
    )
    acc_text = getattr(pl_module, f"{phase}_contrastive_text_accuracy")(
        ret["contrastive_text_logits"], ret["contrastive_labels"]
    )
    pl_module.log(f"contrastive/{phase}/loss", loss)
    pl_module.log(f"contrastive/{phase}/image_accuracy", acc_image)
    pl_module.log(f"contrastive/{phase}/text_accuracy", acc_text)

    if return_infer:
        return ret, infer

    return ret


def compute_cap(pl_module, batch, infer=None, mode="video"):
    if infer is None:  # Skip infer if infer is not None
        infer = pl_module.infer(batch, caption=True, mode=mode)
    cap_logits = infer["cap_logits"]
    # The first is sot_token, prediction starts from the second token
    # Note that there is also an eot_token at the end of each seq
    cap_labels = infer["text_ids"][:, 1:].long()
    special_tokens_mask = infer["special_tokens_mask"][:, 1:]  # 1 for masked

    cap_labels.masked_fill_(special_tokens_mask, value=-100)

    cap_loss = F.cross_entropy(
        cap_logits.reshape(-1, pl_module.hparams.config["vocab_size"]),
        cap_labels.reshape(-1),
        ignore_index=-100,
    )

    ret = {
        "cap_loss": cap_loss,
    }

    phase = "train" if pl_module.training else "val"
    loss = getattr(pl_module, f"{phase}_cap_loss")(cap_loss)
    acc = getattr(pl_module, f"{phase}_cap_accuracy")(
        cap_logits, cap_labels,
    )
    pl_module.log(f"cap/{phase}/loss", loss)
    pl_module.log(f"cap/{phase}/accuracy", acc)

    return ret


def compute_zs_classify(pl_module, batch, text_ret):
    text_feats = text_ret["text_feats"]
    num_text_aug = text_ret["num_text_aug"]
    labels = torch.tensor(batch["answer"]).to(pl_module.device).long()

    video_feats = pl_module.forward_video(batch)["video_feats"]

    if text_feats.ndim == 3:  # [B, N, C] -> [B, C]
        text_feats = text_feats.mean(1)
    if video_feats.ndim == 3:
        video_feats= video_feats.mean(1)

    text_feats /= text_feats.norm(dim=-1, keepdim=True)
    video_feats /= video_feats.norm(dim=-1, keepdim=True)

    similarity = (pl_module.clip.logit_scale.exp() * video_feats @ text_feats.T)
    B, _ = video_feats.shape
    assert similarity.view(B, num_text_aug, -1).shape[-1] == 400, similarity.shape
    similarity = similarity.view(B, num_text_aug, -1).softmax(dim=-1)
    similarity = similarity.mean(dim=1, keepdim=False) 


    phase = "train" if pl_module.training else "val"
    ret = {
        "similarity": similarity,
        "labels": labels,
    }

    acc = getattr(pl_module, f"{phase}_zs_classify_accuracy")(similarity, labels)
    pl_module.log(f"zs_classify/{phase}/accuracy", acc)

    return ret

def compute_mim(pl_module, batch, infer=None, mode="video"):
    if infer is None:  # Skip infer if infer is not None
        infer = pl_module.infer(batch, mask_image=True, mode=mode)
    mim_feats = infer["mim_video_feats"]  # N, Lu, C
    video = infer["video"]  # N, C, T, H, W
    masked_indices = infer["visual_masked_indices"]  # N * T, L

    patch_size = int(math.sqrt(mim_feats.size(-1) // 3 // 2))
    assert 3 * patch_size * patch_size == mim_feats.size(-1)

    if mode == "image":
        assert video.size(3) == 1
        video = video.expand(-1, -1, 2)
    
    img_patch = rearrange(
        video, 'b c (p0 t) (h p1) (w p2) -> b (t h w) (p0 p1 p2) c', 
        p0=2, p1=patch_size, p2=patch_size
    )
    img_patch = (img_patch - img_patch.mean(dim=-2, keepdim=True)
        ) / (img_patch.var(dim=-2, unbiased=True, keepdim=True).sqrt() + 1e-6)
    img_patch = rearrange(img_patch, 'b l p c -> b l (p c)')
    N, _, C = img_patch.shape
    img_patch_mask = img_patch[masked_indices].reshape(N, -1, C)

    mim_loss = F.mse_loss(mim_feats, img_patch_mask)

    ret = {
        "mim_loss": mim_loss,
    }

    phase = "train" if pl_module.training else "val"
    loss = getattr(pl_module, f"{phase}_mim_loss")(ret["mim_loss"])
    pl_module.log(f"mim/{phase}/loss", loss)

    return ret

def init_weights(module):
    if isinstance(module, (nn.Linear, nn.Embedding)):
        module.weight.data.normal_(mean=0.0, std=0.02)
    elif isinstance(module, nn.LayerNorm):
        module.bias.data.zero_()
        module.weight.data.fill_(1.0)

    if isinstance(module, nn.Linear) and module.bias is not None:
        module.bias.data.zero_()


def vqa_test_step(pl_module, batch, output):
    id2answer = (
        pl_module.trainer.datamodule.dm_dicts["vqa_trainval"].id2answer
        if "vqa_trainval" in pl_module.trainer.datamodule.dm_dicts
        else pl_module.trainer.datamodule.dm_dicts["vqa"].id2answer
    )
    vqa_logits = output["vqa_logits"]
    vqa_preds = vqa_logits.argmax(dim=-1)
    vqa_preds = [id2answer[pred.item()] for pred in vqa_preds]
    questions = batch["text"]
    qids = batch["qid"]
    return {"qids": qids, "preds": vqa_preds}


def openend_vqa_test_step(pl_module, batch, output):
    id2answer = (
        pl_module.trainer.datamodule.dm_dicts["vqa_trainval"].id2answer
        if "vqa_trainval" in pl_module.trainer.datamodule.dm_dicts
        else pl_module.trainer.datamodule.dm_dicts["msrvttqa"].id2answer
    )
    vqa_logits = output["vqa_logits"]
    vqa_preds = vqa_logits.argmax(dim=-1)
    vqa_preds = [id2answer[pred.item()] for pred in vqa_preds]
    questions = batch["text"]
    qids = batch["qid"]
    return {"qids": qids, "preds": vqa_preds}


def arc_test_step(pl_module, batch, output):
    return output


def vqa_test_wrapup(outs, model_name):
    rank = torch.distributed.get_rank()
    qids, preds = list(), list()
    for out in outs:
        qids += out["qids"]
        preds += out["preds"]

    rets = list()
    for qid, pred in zip(qids, preds):
        rets.append({"question_id": qid, "answer": pred})
    with open(f"vqa_submit_{rank}.json", "w") as fp:
        json.dump(rets, fp, indent=4)

    torch.distributed.barrier()

    if rank == 0:
        jsons = list()
        paths = list(glob.glob("vqa_submit_*.json"))
        for path in paths:
            with open(path, "r") as fp:
                jsons += json.load(fp)
        os.makedirs("result", exist_ok=True)
        with open(f"result/vqa_submit_{model_name}.json", "w") as fp:
            json.dump(jsons, fp, indent=4)

    torch.distributed.barrier()
    os.remove(f"vqa_submit_{rank}.json")


def arc_test_wrapup(outs, caplen, model_name):
    rank = torch.distributed.get_rank()
    iids, captions = list(), list()
    for out in outs:
        iids += out["iid"]
        captions += out["captions"]

    rets = list()
    for iid, caption in zip(iids, captions):
        rets.append({"video_id": iid, "caption": caption})
    with open(f"coco_cap_len{caplen}_{rank}.json", "w") as fp:
        json.dump(rets, fp, indent=4)

    torch.distributed.barrier()

    if rank == 0:
        jsons = list()
        paths = list(glob.glob(f"coco_cap_len{caplen}_*.json"))
        for path in paths:
            with open(path, "r") as fp:
                jsons += json.load(fp)
        os.makedirs("result/arc", exist_ok=True)
        jsons = sorted(jsons, key=lambda x: x["video_id"])
        with open(f"result/arc/coco_cap_{model_name}_len{caplen}.json", "w") as fp:
            json.dump(jsons, fp, indent=4)

    torch.distributed.barrier()
    os.remove(f"coco_cap_len{caplen}_{rank}.json")
