import torch
import numpy as np


def TubeMaskingGenerator(input_size, mask_ratio, batch, device='cuda'):
    frames, height, width = input_size
    num_patches_per_frame = height * width
    num_masks_per_frame = int(mask_ratio * num_patches_per_frame)

    mask_list = []
    for _ in range(batch):
        mask_per_frame = np.hstack([
            np.zeros(num_patches_per_frame - num_masks_per_frame),
            np.ones(num_masks_per_frame),
        ])
        np.random.shuffle(mask_per_frame)
        mask_list.append(np.tile(mask_per_frame, (frames, 1)).flatten())
    mask = torch.Tensor(mask_list).to(device, non_blocking=True).to(torch.bool)
    return mask 


def RandomMaskingGenerator(input_size, mask_ratio, batch, device='cuda'):
    frames, height, width = input_size

    num_patches = frames * height * width  # 8x14x14
    num_mask = int(mask_ratio * num_patches)

    mask_list = []
    for _ in range(batch):
        mask = np.hstack([
            np.zeros(num_patches - num_mask),
            np.ones(num_mask),
        ])
        np.random.shuffle(mask)
        mask_list.append(mask)
    mask = torch.Tensor(mask_list).to(device, non_blocking=True).to(torch.bool)
    return mask


def DoubleRandomMaskingGenerator(input_size, mask_ratio, double_mask_ratio, batch, device='cuda'):
    frames, height, width = input_size
    num_patches_per_frame = height * width
    num_patches = frames * num_patches_per_frame 
    num_masks_per_frame = int(mask_ratio * num_patches_per_frame)
    num_mask = frames * num_masks_per_frame
    
    num_vis = num_patches - num_mask
    num_double_mask = int(num_vis * double_mask_ratio)

    mask_list = []
    double_mask_list = []
    for _ in range(batch):
        mask_per_frame = np.hstack([
            np.zeros(num_patches_per_frame - num_masks_per_frame),
            np.ones(num_masks_per_frame),
        ])
        np.random.shuffle(mask_per_frame)
        mask = np.tile(mask_per_frame, (frames, 1)).flatten()
        
        double_mask = np.hstack([
            np.zeros(num_vis - num_double_mask),
            np.ones(num_double_mask),
        ])
        np.random.shuffle(double_mask)

        mask_list.append(mask)
        double_mask_list.append(double_mask_list)

    mask = torch.Tensor(mask_list).to(device, non_blocking=True).to(torch.bool)
    double_mask = torch.Tensor(double_mask_list).to(device, non_blocking=True).to(torch.bool)
    return mask. double_mask