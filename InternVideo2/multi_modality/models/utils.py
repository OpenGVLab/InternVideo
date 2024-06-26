import logging

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
#from scipy import interpolate
from typing import List

from torch import nn

logger = logging.getLogger(__name__)


def load_temp_embed_with_mismatch(temp_embed_old, temp_embed_new, add_zero=True):
    """
    Add/Remove extra temporal_embeddings as needed.
    https://arxiv.org/abs/2104.00650 shows adding zero paddings works.

    temp_embed_old: (1, num_frames_old, 1, d)
    temp_embed_new: (1, num_frames_new, 1, d)
    add_zero: bool, if True, add zero, else, interpolate trained embeddings.
    """
    # TODO zero pad
    num_frms_new = temp_embed_new.shape[1]
    num_frms_old = temp_embed_old.shape[1]
    logger.info(f"Load temporal_embeddings, lengths: {num_frms_old}-->{num_frms_new}")
    if num_frms_new > num_frms_old:
        if add_zero:
            temp_embed_new[
                :, :num_frms_old
            ] = temp_embed_old  # untrained embeddings are zeros.
        else:
            temp_embed_new = interpolate_temporal_pos_embed(temp_embed_old, num_frms_new)
    elif num_frms_new < num_frms_old:
        temp_embed_new = temp_embed_old[:, :num_frms_new]
    else:  # =
        temp_embed_new = temp_embed_old
    return temp_embed_new


def interpolate_temporal_pos_embed(temp_embed_old, num_frames_new):
    """
    temp_embed_old: (1, num_frames_old, 1, d)
    Returns:
        temp_embed_new: (1, num_frames_new, 1, d)
    """
    temp_embed_old = temp_embed_old.squeeze(2).permute(
        0, 2, 1
    )  # (1, d, num_frames_old)
    temp_embed_new = F.interpolate(
        temp_embed_old, num_frames_new, mode="linear"
    )  # (1, d, num_frames_new)
    temp_embed_new = temp_embed_new.permute(0, 2, 1).unsqueeze(
        2
    )  # (1, num_frames_new, 1, d)
    return temp_embed_new


def interpolate_pos_embed(pos_embed_old, pos_embed_new, num_patches_new):
    """
    Args:
        pos_embed_old: (1, L_old, d), pre-trained
        pos_embed_new: (1, L_new, d), newly initialized, to be replaced by interpolated weights
        num_patches_new:
    """
    # interpolate position embedding
    embedding_size = pos_embed_old.shape[-1]
    num_extra_tokens = pos_embed_new.shape[-2] - num_patches_new
    # height (== width) for the checkpoint position embedding
    orig_size = int((pos_embed_old.shape[-2] - num_extra_tokens) ** 0.5)
    # height (== width) for the new position embedding
    new_size = int(num_patches_new ** 0.5)

    if orig_size != new_size:
        # class_token and dist_token are kept unchanged
        # the extra tokens seems always at the beginning of the position embedding
        extra_tokens = pos_embed_old[:, :num_extra_tokens]
        # only the position tokens are interpolated
        pos_tokens = pos_embed_old[:, num_extra_tokens:]
        pos_tokens = pos_tokens.reshape(
            -1, orig_size, orig_size, embedding_size
        ).permute(0, 3, 1, 2)
        pos_tokens = torch.nn.functional.interpolate(
            pos_tokens, size=(new_size, new_size), mode="bicubic", align_corners=False
        )
        pos_tokens = pos_tokens.permute(0, 2, 3, 1).flatten(1, 2)
        interpolated_pos_embed = torch.cat((extra_tokens, pos_tokens), dim=1)
        logger.info(f"reshape position embedding from {orig_size}**2 to {new_size}**2")
        return interpolated_pos_embed
    else:
        return pos_embed_old


def interpolate_pos_relative_bias_beit(state_dict_old, state_dict_new, patch_shape_new):
    """
    Args:
        state_dict_old: loaded state dict
        state_dict_new: state dict for model with new image size
        patch_shape_new: new model patch_shape
    ref: https://github.com/microsoft/unilm/blob/master/beit/run_class_finetuning.py
    """
    all_keys = list(state_dict_old.keys())
    for key in all_keys:
        if "relative_position_index" in key:
            state_dict_old.pop(key)

        if "relative_position_bias_table" in key:
            rel_pos_bias = state_dict_old[key]
            src_num_pos, num_attn_heads = rel_pos_bias.size()
            dst_num_pos, _ = state_dict_new[key].size()
            dst_patch_shape = patch_shape_new
            if dst_patch_shape[0] != dst_patch_shape[1]:
                raise NotImplementedError()
            num_extra_tokens = dst_num_pos - (dst_patch_shape[0] * 2 - 1) * (
                dst_patch_shape[1] * 2 - 1
            )
            src_size = int((src_num_pos - num_extra_tokens) ** 0.5)
            dst_size = int((dst_num_pos - num_extra_tokens) ** 0.5)
            if src_size != dst_size:
                # logger.info("Position interpolate for %s from %dx%d to %dx%d" % (
                #     key, src_size, src_size, dst_size, dst_size))
                extra_tokens = rel_pos_bias[-num_extra_tokens:, :]
                rel_pos_bias = rel_pos_bias[:-num_extra_tokens, :]

                def geometric_progression(a, r, n):
                    return a * (1.0 - r ** n) / (1.0 - r)

                left, right = 1.01, 1.5
                while right - left > 1e-6:
                    q = (left + right) / 2.0
                    gp = geometric_progression(1, q, src_size // 2)
                    if gp > dst_size // 2:
                        right = q
                    else:
                        left = q

                # if q > 1.090307:
                #     q = 1.090307

                dis = []
                cur = 1
                for i in range(src_size // 2):
                    dis.append(cur)
                    cur += q ** (i + 1)

                r_ids = [-_ for _ in reversed(dis)]

                x = r_ids + [0] + dis
                y = r_ids + [0] + dis

                t = dst_size // 2.0
                dx = np.arange(-t, t + 0.1, 1.0)
                dy = np.arange(-t, t + 0.1, 1.0)

                # logger.info("Original positions = %s" % str(x))
                # logger.info("Target positions = %s" % str(dx))

                all_rel_pos_bias = []

                for i in range(num_attn_heads):
                    z = rel_pos_bias[:, i].view(src_size, src_size).float().numpy()
                    f = interpolate.interp2d(x, y, z, kind="cubic")
                    all_rel_pos_bias.append(
                        torch.Tensor(f(dx, dy))
                        .contiguous()
                        .view(-1, 1)
                        .to(rel_pos_bias.device)
                    )

                rel_pos_bias = torch.cat(all_rel_pos_bias, dim=-1)

                new_rel_pos_bias = torch.cat((rel_pos_bias, extra_tokens), dim=0)
                state_dict_old[key] = new_rel_pos_bias
    return state_dict_old


def tile(x, dim, n_tile):
    init_dim = x.size(dim)
    repeat_idx = [1] * x.dim()
    repeat_idx[dim] = n_tile
    x = x.repeat(*repeat_idx)
    order_index = torch.LongTensor(
        np.concatenate([init_dim * np.arange(n_tile) + i for i in range(init_dim)])
    )
    return torch.index_select(x, dim, order_index.to(x.device))


def mask_logits(target, mask):
    return target * mask + (1 - mask) * (-1e10)


class AllGather(torch.autograd.Function):
    """An autograd function that performs allgather on a tensor."""

    @staticmethod
    def forward(ctx, tensor, args):
        output = [torch.empty_like(tensor) for _ in range(args.world_size)]
        torch.distributed.all_gather(output, tensor)
        ctx.rank = args.rank
        ctx.batch_size = tensor.shape[0]
        return torch.cat(output, dim=0)

    @staticmethod
    def backward(ctx, grad_output):
        return (
            grad_output[ctx.batch_size * ctx.rank : ctx.batch_size * (ctx.rank + 1)],
            None,
        )


allgather_wgrad = AllGather.apply


def tie_encoder_decoder_weights(
    encoder: nn.Module, decoder: nn.Module, base_model_prefix: str, skip_key: str
):
    uninitialized_encoder_weights: List[str] = []
    if decoder.__class__ != encoder.__class__:
        if issubclass(decoder.__class__, encoder.__class__):
            logger.info(
                f"decoder ({decoder.__class__}) and encoder ({encoder.__class__}) are not equal, encoder is decoder's father. In this case make sure that all encoder weights are correctly initialized."
            )
        elif issubclass(encoder.__class__, decoder.__class__):
            logger.info(
                f"decoder ({decoder.__class__}) and encoder ({encoder.__class__}) are not equal, decoder is encoder's father. In this case make sure that all encoder weights are correctly initialized."
            )
        else:
            raise ValueError(f"decoder ({decoder.__class__}) and encoder ({encoder.__class__}) are not equal!!!")

    def tie_encoder_to_decoder_recursively(
        decoder_pointer: nn.Module,
        encoder_pointer: nn.Module,
        module_name: str,
        uninitialized_encoder_weights: List[str],
        skip_key: str,
        depth=0,
    ):
        assert isinstance(decoder_pointer, nn.Module) and isinstance(
            encoder_pointer, nn.Module
        ), f"{decoder_pointer} and {encoder_pointer} have to be of type torch.nn.Module"
        if hasattr(decoder_pointer, "weight") and skip_key not in module_name:
            assert hasattr(encoder_pointer, "weight")
            encoder_pointer.weight = decoder_pointer.weight
            if hasattr(decoder_pointer, "bias"):
                assert hasattr(encoder_pointer, "bias")
                encoder_pointer.bias = decoder_pointer.bias
            logger.info(module_name + " is tied")
            return

        encoder_modules = encoder_pointer._modules
        decoder_modules = decoder_pointer._modules
        if len(decoder_modules) > 0:
            assert (
                len(encoder_modules) > 0
            ), f"Encoder module {encoder_pointer} does not match decoder module {decoder_pointer}"

            all_encoder_weights = set(
                [module_name + "/" + sub_name for sub_name in encoder_modules.keys()]
            )
            encoder_layer_pos = 0
            for name, module in decoder_modules.items():
                if name.isdigit():
                    encoder_name = str(int(name) + encoder_layer_pos)
                    decoder_name = name
                    if not isinstance(
                        decoder_modules[decoder_name],
                        type(encoder_modules[encoder_name]),
                    ) and len(encoder_modules) != len(decoder_modules):
                        # this can happen if the name corresponds to the position in a list module list of layers
                        # in this case the decoder has added a cross-attention that the encoder does not have
                        # thus skip this step and subtract one layer pos from encoder
                        encoder_layer_pos -= 1
                        continue
                elif name not in encoder_modules:
                    continue
                elif depth > 500:
                    raise ValueError(
                        "Max depth of recursive function `tie_encoder_to_decoder` reached. It seems that there is a circular dependency between two or more `nn.Modules` of your model."
                    )
                else:
                    decoder_name = encoder_name = name
                tie_encoder_to_decoder_recursively(
                    decoder_modules[decoder_name],
                    encoder_modules[encoder_name],
                    module_name + "/" + name,
                    uninitialized_encoder_weights,
                    skip_key,
                    depth=depth + 1,
                )
                all_encoder_weights.remove(module_name + "/" + encoder_name)

            uninitialized_encoder_weights += list(all_encoder_weights)

    # tie weights recursively
    tie_encoder_to_decoder_recursively(
        decoder, encoder, base_model_prefix, uninitialized_encoder_weights, skip_key
    )
