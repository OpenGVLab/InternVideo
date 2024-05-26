"""
Non-Maximum Suppression for video proposals.
"""


def compute_temporal_iou(pred, gt):
    """ deprecated due to performance concerns
    compute intersection-over-union along temporal axis
    Args:
        pred: [st (float), ed (float)]
        gt: [st (float), ed (float)]
    Returns:
        iou (float):

    Ref: https://github.com/LisaAnne/LocalizingMoments/blob/master/utils/eval.py
    """
    intersection = max(0, min(pred[1], gt[1]) - max(pred[0], gt[0]))
    union = max(pred[1], gt[1]) - min(pred[0], gt[0])  # not the correct union though
    if union == 0:
        return 0
    else:
        return 1.0 * intersection / union


def temporal_nms(predictions, nms_thd, max_after_nms=100):
    """
    Args:
        predictions: list(sublist), each sublist is [st (float), ed(float), score (float)],
            note larger scores are better and are preserved. For metrics that are better when smaller,
            please convert to its negative, e.g., convert distance to negative distance.
        nms_thd: float in [0, 1]
        max_after_nms:
    Returns:
        predictions_after_nms: list(sublist), each sublist is [st (float), ed(float), score (float)]
    References:
        https://github.com/wzmsltw/BSN-boundary-sensitive-network/blob/7b101fc5978802aa3c95ba5779eb54151c6173c6/Post_processing.py#L42
    """
    if len(predictions) == 1:  # only has one prediction, no need for nms
        return predictions

    predictions = sorted(predictions, key=lambda x: x[2], reverse=True)  # descending order

    tstart = [e[0] for e in predictions]
    tend = [e[1] for e in predictions]
    tscore = [e[2] for e in predictions]
    rstart = []
    rend = []
    rscore = []
    while len(tstart) > 1 and len(rscore) < max_after_nms:  # max 100 after nms
        idx = 1
        while idx < len(tstart):  # compare with every prediction in the list.
            if compute_temporal_iou([tstart[0], tend[0]], [tstart[idx], tend[idx]]) > nms_thd:
                # rm highly overlapped lower score entries.
                tstart.pop(idx)
                tend.pop(idx)
                tscore.pop(idx)
                # print("--------------------------------")
                # print(compute_temporal_iou([tstart[0], tend[0]], [tstart[idx], tend[idx]]))
                # print([tstart[0], tend[0]], [tstart[idx], tend[idx]])
                # print(tstart.pop(idx), tend.pop(idx), tscore.pop(idx))
            else:
                # move to next
                idx += 1
        rstart.append(tstart.pop(0))
        rend.append(tend.pop(0))
        rscore.append(tscore.pop(0))

    if len(rscore) < max_after_nms and len(tstart) >= 1:  # add the last, possibly empty.
        rstart.append(tstart.pop(0))
        rend.append(tend.pop(0))
        rscore.append(tscore.pop(0))

    predictions_after_nms = [[st, ed, s] for s, st, ed in zip(rscore, rstart, rend)]
    return predictions_after_nms
