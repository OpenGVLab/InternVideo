import os
import shutil
import time
import json
import pickle
from typing import Dict

import numpy as np
import pdb
import torch
from scipy.special import softmax
from .metrics import ANETdetection


# def load_results_from_pkl(filename):
#     # load from pickle file
#     assert os.path.isfile(filename)
#     with open(filename, "rb") as f:
#         results = pickle.load(f)
#     return results

def load_results_from_json(filename):
    assert os.path.isfile(filename)
    with open(filename, "r") as f:
        results = json.load(f)
    # for activity net external classification scores
    if 'results' in results:
        results = results['results']
    return results

def results_to_dict(results):
    """convert result arrays into dict used by json files"""
    # video ids and allocate the dict
    vidxs = sorted(list(set(results['video-id'])))
    results_dict = {}
    for vidx in vidxs:
        results_dict[vidx] = []

    # fill in the dict
    for vidx, start, end, label, score in zip(
        results['video-id'],
        results['t-start'],
        results['t-end'],
        results['label'],
        results['score']
    ):
        results_dict[vidx].append(
            {
                "label" : int(label),
                "score" : float(score),
                "segment": [float(start), float(end)],
            }
        )
    return results_dict


def results_to_array(results, num_pred):
    # video ids and allocate the dict
    vidxs = sorted(list(set(results['video-id'])))
    results_dict = {}
    for vidx in vidxs:
        results_dict[vidx] = {
            'label'   : [],
            'score'   : [],
            'segment' : [],
        }

    # fill in the dict
    for vidx, start, end, label, score in zip(
        results['video-id'],
        results['t-start'],
        results['t-end'],
        results['label'],
        results['score']
    ):
        results_dict[vidx]['label'].append(int(label))
        results_dict[vidx]['score'].append(float(score))
        results_dict[vidx]['segment'].append(
            [float(start), float(end)]
        )

    for vidx in vidxs:
        label = np.asarray(results_dict[vidx]['label'])
        score = np.asarray(results_dict[vidx]['score'])
        segment = np.asarray(results_dict[vidx]['segment'])

        # the score should be already sorted, just for safety
        inds = np.argsort(score)[::-1][:num_pred]
        label, score, segment = label[inds], score[inds], segment[inds]
        results_dict[vidx]['label'] = label
        results_dict[vidx]['score'] = score
        results_dict[vidx]['segment'] = segment

    return results_dict


def postprocess_results(results, cls_score_file, num_pred=200, topk=2):

    # load results and convert to dict
    # if isinstance(results, str):
    #     results = load_results_from_pkl(results)
    # array -> dict
    results = results_to_array(results, num_pred)

    # load external classification scores
    if '.json' in cls_score_file:
        cls_scores = load_results_from_json(cls_score_file)
    # else:
    #     cls_scores = load_results_from_pkl(cls_score_file)

    # dict for processed results
    processed_results = {
        'video-id': [],
        't-start' : [],
        't-end': [],
        'label': [],
        'score': []
    }

    # process each video
    for vid, result in results.items():

        # pick top k cls scores and idx
        if len(cls_scores[vid])==1:
            curr_cls_scores = np.asarray(cls_scores[vid][0])
        else:
            curr_cls_scores = np.asarray(cls_scores[vid])

        if max(curr_cls_scores)>1 or min(curr_cls_scores)<0:
            curr_cls_scores=softmax(curr_cls_scores)
        
        topk_cls_idx = np.argsort(curr_cls_scores)[::-1][:topk]
        topk_cls_score = curr_cls_scores[topk_cls_idx]

        # model outputs
        pred_score, pred_segment, pred_label = \
            result['score'], result['segment'], result['label']
        num_segs = min(num_pred, len(pred_score))

        # duplicate all segment and assign the topk labels
        # K x 1 @ 1 N -> K x N -> KN
        # multiply the scores
        # temp = np.abs(topk_cls_score[:, None] @ pred_score[None, :])
        # new_pred_score = np.sqrt(temp).flatten()        
        new_pred_score = np.sqrt(topk_cls_score[:, None] @ pred_score[None, :]).flatten()
        new_pred_segment = np.tile(pred_segment, (topk, 1))
        new_pred_label = np.tile(topk_cls_idx[:, None], (1, num_segs)).flatten()

        # add to result
        processed_results['video-id'].extend([vid]*num_segs*topk)
        processed_results['t-start'].append(new_pred_segment[:, 0])
        processed_results['t-end'].append(new_pred_segment[:, 1])
        processed_results['label'].append(new_pred_label)
        processed_results['score'].append(new_pred_score)
        # pdb.set_trace()

    processed_results['t-start'] = np.concatenate(
        processed_results['t-start'], axis=0)
    processed_results['t-end'] = np.concatenate(
        processed_results['t-end'], axis=0)
    processed_results['label'] = np.concatenate(
        processed_results['label'],axis=0)
    processed_results['score'] = np.concatenate(
        processed_results['score'], axis=0)

    return processed_results
