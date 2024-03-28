# --------------------------------------------------------
# Based on BEiT, timm, DINO and DeiT code bases
# https://github.com/microsoft/unilm/tree/master/beit
# https://github.com/rwightman/pytorch-image-models/tree/master/timm
# https://github.com/facebookresearch/deit
# https://github.com/facebookresearch/dino
# --------------------------------------------------------'
import math
import random

import numpy as np
import torch
from einops import rearrange


def topk(matrix, K, axis=1):
    if axis == 0:
        row_index = np.arange(matrix.shape[1 - axis])
        topk_index = np.argpartition(-matrix, K, axis=axis)[0:K, :]
        topk_data = matrix[topk_index, row_index]
        topk_index_sort = np.argsort(-topk_data, axis=axis)
        topk_data_sort = topk_data[topk_index_sort, row_index]
        topk_index_sort = topk_index[0:K, :][topk_index_sort, row_index]
    else:
        column_index = np.arange(matrix.shape[1 - axis])[:, None]
        topk_index = np.argpartition(-matrix, K, axis=axis)[:, 0:K]
        topk_data = matrix[column_index, topk_index]
        topk_index_sort = np.argsort(-topk_data, axis=axis)
        topk_data_sort = topk_data[column_index, topk_index_sort]
        topk_index_sort = topk_index[:, 0:K][column_index, topk_index_sort]
    return (topk_data_sort, topk_index_sort)


class MaskingGenerator:

    def update_state(self, epoch):
        pass


class RandomMaskingGenerator(MaskingGenerator):

    def __init__(self, input_size, mask_ratio):
        if not isinstance(input_size, tuple):
            input_size = (input_size, ) * 3

        self.frames, self.height, self.width = input_size

        self.num_patches = self.frames * self.height * self.width  # 8x14x14
        self.num_mask = int(mask_ratio * self.num_patches)

    def __repr__(self):
        repr_str = "Mask: total patches {}, mask patches {}".format(
            self.num_patches, self.num_mask)
        return repr_str

    def __call__(self):
        mask = np.hstack([
            np.zeros(self.num_patches - self.num_mask),
            np.ones(self.num_mask),
        ])
        np.random.shuffle(mask)
        return mask  # [196*8]


class Cell():

    def __init__(self, num_masks, num_patches):
        self.num_masks = num_masks
        self.num_patches = num_patches
        self.size = num_masks + num_patches
        self.queue = np.hstack([np.ones(num_masks), np.zeros(num_patches)])
        self.queue_ptr = 0

    def set_ptr(self, pos=-1):
        self.queue_ptr = np.random.randint(self.size) if pos < 0 else pos

    def get_cell(self):
        cell_idx = (np.arange(self.size) + self.queue_ptr) % self.size
        return self.queue[cell_idx]

    def run_cell(self):
        self.queue_ptr += 1


class CellRunningMaskingGenerator(MaskingGenerator):

    def __init__(self, input_size, mask_ratio=0.5, is_train=True):
        self.frames, self.height, self.width = input_size
        self.mask_ratio = mask_ratio
        self.ptr_pos = -1 if is_train else 0

        num_masks_per_cell = int(4 * self.mask_ratio)
        assert 0 < num_masks_per_cell < 4
        num_patches_per_cell = 4 - num_masks_per_cell

        self.cell = Cell(num_masks_per_cell, num_patches_per_cell)
        self.cell_size = self.cell.size

        mask_list = []
        for ptr_pos in range(self.cell_size):
            self.cell.set_ptr(ptr_pos)
            mask = []
            for _ in range(self.frames):
                self.cell.run_cell()
                mask_unit = self.cell.get_cell().reshape(2, 2)
                mask_map = np.tile(mask_unit,
                                   [self.height // 2, self.width // 2])
                mask.append(mask_map)
            mask = np.stack(mask, axis=0).flatten()
            mask_list.append(mask)
        self.all_mask_maps = np.stack(mask_list, axis=0)

    def __repr__(self):
        repr_str = f"Cell Running Mask with mask ratio {self.mask_ratio}"
        return repr_str

    def __call__(self, batch_size):
        mask_idx_list = np.random.randint(self.cell_size, size=(batch_size))
        return torch.as_tensor(self.all_mask_maps[mask_idx_list])


class RandomDecodeMaskingGenerator(MaskingGenerator):

    def __init__(self, input_size, mask_ratio=0.5):
        self.frame, self.height, self.width = input_size
        self.mask_ratio = mask_ratio

        self.num_patches = self.frame * self.height * self.width  # 8x14x14
        self.num_mask = int(mask_ratio * self.num_patches)

    def __repr__(self):
        repr_str = "Mask: total patches {}, mask patches {}".format(
            self.num_patches, self.num_mask)
        return repr_str

    def __call__(self, batch_size):
        rand = torch.as_tensor(np.random.randn(batch_size, self.num_patches))
        mask_idx = torch.topk(rand, self.num_mask, dim=-1,
                              sorted=False).indices
        mask = torch.zeros(batch_size,
                           self.num_patches).scatter_(-1, mask_idx, 1)
        return mask


class TemporalConsistencyMaskingGenerator(MaskingGenerator):

    def __init__(self, input_size, mask_ratio):
        self.frames, self.height, self.width = input_size
        self.num_patches_per_frame = self.height * self.width  # 14x14
        self.total_patches = self.frames * self.num_patches_per_frame
        self.num_masks_per_frame = int(mask_ratio * self.num_patches_per_frame)
        self.total_masks = self.frames * self.num_masks_per_frame

    def __repr__(self):
        repr_str = "Mask: total patches {}, mask patches {}".format(
            self.total_patches, self.total_masks)
        return repr_str

    def __call__(self):
        mask_per_frame = np.hstack([
            np.zeros(self.num_patches_per_frame - self.num_masks_per_frame),
            np.ones(self.num_masks_per_frame),
        ])
        np.random.shuffle(mask_per_frame)
        mask = np.tile(mask_per_frame, (self.frames, 1)).flatten()
        return mask  # [196*8]


class TemporalProgressiveMaskingGenerator(MaskingGenerator):

    def __init__(self, input_size, mask_ratio):
        self.frames, self.height, self.width = input_size
        self.num_patches_per_frame = self.height * self.width  # 14x14
        self.total_patches = self.frames * self.num_patches_per_frame  # 8x14x14
        max_keep_patch = int(
            (1 - mask_ratio) * self.num_patches_per_frame)  # 1 - 0.75 = 0.25
        min_keep_patch = int(0.05 * self.num_patches_per_frame)
        self.keep_patches_list = np.linspace(max_keep_patch, min_keep_patch,
                                             self.frames).astype(int)
        self.total_masks = self.total_patches - self.keep_patches_list.sum()

    def __repr__(self):
        repr_str = "Mask: total patches {}, mask patches {}".format(
            self.total_patches, self.total_masks)
        return repr_str

    def __call__(self):

        rand = np.random.randn(1, self.num_patches_per_frame)
        mask = np.zeros((self.frames, self.num_patches_per_frame),
                        dtype=np.bool)
        for i in range(self.frames):
            top_k, _ = topk(rand, self.keep_patches_list[i])
            the_topk = top_k[0][-1]
            mask[i] = rand <= the_topk
        mask = mask.flatten().astype(int)
        return mask  # [196*8]


class TemporalCenteringProgressiveMaskingGenerator(MaskingGenerator):

    def __init__(self, input_size, mask_ratio):
        self.num_frames, self.height, self.width = input_size
        self.num_patches_per_frame = self.height * self.width  # 14x14
        self.total_patches = self.num_frames * self.num_patches_per_frame  # 8x14x14
        min_mask_ratio = mask_ratio  # 0.9 -> keep 19 token
        # 0.979 -> keep 4 token  0.95 -> keep 9 token
        max_mask_ratio = 0.95
        max_keep_patch = int(
            (1 - min_mask_ratio) * self.num_patches_per_frame)  # 1 - 0.9 = 0.1
        min_keep_patch = int((1 - max_mask_ratio) *
                             self.num_patches_per_frame)  # 1 - 0.95 = 0.05
        patches_list = np.linspace(max_keep_patch, min_keep_patch,
                                   self.num_frames // 2).astype(int).tolist()
        self.keep_patches_list = patches_list.copy()
        patches_list.reverse()
        self.keep_patches_list = patches_list + self.keep_patches_list
        self.total_masks = self.total_patches - sum(self.keep_patches_list)

    def __repr__(self):
        repr_str = "Mask: total patches {}, mask patches {}".format(
            self.total_patches, self.total_masks)
        return repr_str

    def __call__(self):

        rand = np.random.randn(1, self.num_patches_per_frame)
        mask = np.zeros((self.num_frames, self.num_patches_per_frame),
                        dtype=np.bool)
        for i in range(self.num_frames):
            top_k, _ = topk(rand, self.keep_patches_list[i])
            the_topk = top_k[0][-1]
            mask[i] = rand <= the_topk
        mask = mask.flatten().astype(int)
        return mask  # [196*8]
