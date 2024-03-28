import torch
import numpy as np
import decord

from typing import Any, OrderedDict, Union, List
from pkg_resources import packaging
from torchvision import transforms

from . import video_transform
from .simple_tokenizer import SimpleTokenizer as _Tokenizer
from .clip_utils.model import build_model
from .clip_utils import load as load


__all__ = ["load_model", "load_video", "tokenize", "load"]
_tokenizer = _Tokenizer()


def load_model(path):
    state = torch.load(path, map_location="cpu")["state_dict"]
    state = {k[len("clip.") :]: v for k, v in state.items() if k.startswith("clip.")}
    model = build_model(state_dict=state)
    return model


def load_video(path):
    video_reader = decord.VideoReader(path, num_threads=1, ctx=decord.cpu(0))
    decord.bridge.set_bridge('torch')
    video_len = len(video_reader)
    video = video_reader.get_batch(np.linspace(0, video_len - 1, 8).astype(np.int)).byte()
    video = video.permute(3, 0, 1, 2)

    input_mean = [0.48145466, 0.4578275, 0.40821073]
    input_std = [0.26862954, 0.26130258, 0.27577711]
    crop_size, scale_size = 224, 256
    trans = transforms.Compose([
        video_transform.TensorToNumpy(),
        video_transform.Resize(scale_size),
        video_transform.CenterCrop(crop_size),
        video_transform.ClipToTensor(channel_nb=3),
        video_transform.Normalize(mean=input_mean, std=input_std)
    ])

    video = trans(video)
    return video


def tokenize(
    texts: Union[str, List[str]],
    context_length: int = 77,
    truncate: bool = False,
    return_special_tokens_mask: bool = False,
) -> Union[torch.IntTensor, torch.LongTensor, torch.BoolTensor]:
    """
    Returns the tokenized representation of given input string(s)

    Parameters
    ----------
    texts : Union[str, List[str]]
        An input string or a list of input strings to tokenize

    context_length : int
        The context length to use; all CLIP models use 77 as the context length

    truncate: bool
        Whether to truncate the text in case its encoding is longer than the context length

    Returns
    -------
    A two-dimensional tensor containing the resulting tokens, shape = [number of input strings, context_length].
    We return LongTensor when torch version is <1.8.0, since older index_select requires indices to be long.
    """
    if isinstance(texts, str):
        texts = [texts]

    sot_token = _tokenizer.encoder["<|startoftext|>"]
    eot_token = _tokenizer.encoder["<|endoftext|>"]
    all_tokens = [[sot_token] + _tokenizer.encode(text) + [eot_token] for text in texts]
    if packaging.version.parse(torch.__version__) < packaging.version.parse("1.8.0"):
        result = torch.zeros(len(all_tokens), context_length, dtype=torch.long)
    else:
        result = torch.zeros(len(all_tokens), context_length, dtype=torch.int)

    special_tokens_mask = torch.zeros(len(all_tokens), context_length, dtype=torch.bool)

    for i, tokens in enumerate(all_tokens):
        if len(tokens) > context_length:
            if truncate:
                tokens = tokens[:context_length]
                tokens[-1] = eot_token
            else:
                raise RuntimeError(
                    f"Input {texts[i]} is too long for context length {context_length}"
                )
        result[i, : len(tokens)] = torch.tensor(tokens)
        special_tokens_mask[i, len(tokens) :] = 1

    if return_special_tokens_mask:
        return result, special_tokens_mask

    return result
