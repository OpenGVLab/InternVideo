import logging
import torch
import pickle
from collections import OrderedDict


def _rename_weights(weights, weight_map):
    logger = logging.getLogger(__name__)
    logger.info("Remapping C2 weights")
    max_c2_key_size = max([len(k) for k in weight_map.values()])
    new_weights = OrderedDict()
    for k in weight_map:
        c2_name = weight_map[k]
        logger.info("C2 name: {: <{}} mapped name: {}".format(c2_name, max_c2_key_size, k))
        if c2_name not in weights:
            logger.info("{} not found in C2 weights file, skipped.".format(c2_name))
            continue
        v = weights[c2_name]
        w = torch.from_numpy(v)
        new_weights[k] = w
    return new_weights


def _load_c2_pickled_weights(file_path):
    with open(file_path, "rb") as f:
        if torch._six.PY3:
            data = pickle.load(f, encoding="latin1")
        else:
            data = pickle.load(f)
    if "blobs" in data:
        weights = data["blobs"]
    else:
        weights = data
    return weights


def load_c2_format(f, weight_map):
    # We also support load from caffe2 weights.
    state_dict = _load_c2_pickled_weights(f)
    state_dict = _rename_weights(state_dict, weight_map)
    return dict(model=state_dict)
