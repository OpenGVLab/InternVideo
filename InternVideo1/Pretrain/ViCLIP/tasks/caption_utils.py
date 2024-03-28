import datetime
import logging
import time
import itertools

import torch
import torch.distributed as dist

from utils.basic_utils import MetricLogger

from pycocoevalcap.tokenizer.ptbtokenizer import PTBTokenizer
from pycocoevalcap.bleu.bleu import Bleu
from pycocoevalcap.cider.cider import Cider
from pycocoevalcap.meteor.meteor import Meteor
from pycocoevalcap.rouge.rouge import Rouge
from pycocoevalcap.spice.spice import Spice

from utils.basic_utils import MetricLogger

logger = logging.getLogger(__name__)


def eval_nlp_scores(pred, gt, verbose=False):
    """
    Stolen from https://github.com/zohrehghaderi/VASTA/blob/ede0461fd0fc00da575bfca2399532e8eb7607ac/nlp_metrics/cocval_evalution.py#L9
    evaluates the nlp scores bleu1-bleu4, meteor, rouge-l, cider, spice
    Also logs the corpus values as scalars and the individual scores as histograms!
    Args:
        pred (List): List of predictions
        gt (List): List of ground truths
    """
    tokenizer = PTBTokenizer()
    
    gts = tokenizer.tokenize(gt)
    res = tokenizer.tokenize(pred)

    # Set up scorers
    if verbose: print('Setting up scorers...')
    scorers = [
        (Bleu(4), ["Bleu_1", "Bleu_2", "Bleu_3", "Bleu_4"]),
        (Meteor(), "METEOR"),
        (Rouge(), "ROUGE_L"),
        (Cider(), "CIDEr")
    ]

    # Compute scores
    results = {}
    for scorer, method in scorers:
        score, scores= scorer.compute_score(gts, res)
        if isinstance(method, list):
            for sc, scs, m in zip(score, scores, method):
                results[m] = sc
        else:
            results[method] = score

    return results

@torch.no_grad()
def evaluation_wrapper(model, data_loader, tokenizer, device, config, prefix=""):
    model.eval()

    metric_logger = MetricLogger(delimiter="  ")
    header = "[evaluation] Generating captions:"
    log_freq = config.log_freq // 2

    logger.info("Start generating results.")

    iterator = metric_logger.log_every(data_loader, log_freq, header)

    all_texts = data_loader.dataset.text
    img2txt = data_loader.dataset.img2txt
    for v in img2txt.values():
        assert len(v) == 1, "Only support one caption per image"
    img2txt = {k: v[0] for k, v in img2txt.items()}

    all_pred_caption = []
    all_gt_caption = []

    for n, (image, idx) in enumerate(iterator):
        image = image.to(device, non_blocking=True)
        caption = [all_texts[img2txt[idx[i].item()]] for i in range(len(idx))]

        pred_tokens, pred_caption = model(
            image,
            train=False,
            raw_caption=caption,
        )

        all_pred_caption += pred_caption
        all_gt_caption += caption
    
    logger.info("Finish generating results.")
    logger.info("Computing accuracy.")
    
    preds = [None] * dist.get_world_size()
    gts = [None] * dist.get_world_size()
    dist.all_gather_object(preds, all_pred_caption)
    dist.all_gather_object(gts, all_gt_caption)

    preds = list(itertools.chain(*preds))
    gts = list(itertools.chain(*gts))

    preds = {k: [{'caption': v}] for k, v in enumerate(preds)}
    gts = {k: [{'caption': v}] for k, v in enumerate(gts)}

    if dist.get_rank() == 0:
        results = eval_nlp_scores(preds, gts, verbose=False)
        results = [results]
    else:
        results = [None]
    
    dist.broadcast_object_list(results, src=0)
    
    return {prefix: results[0]}

