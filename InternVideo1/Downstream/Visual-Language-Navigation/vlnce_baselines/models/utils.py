import math
from turtle import heading
import torch


def angle_feature(headings, device=None):
    # twopi = math.pi * 2
    # heading = (heading + twopi) % twopi     # From 0 ~ 2pi
    # It will be the same
    heading_enc = torch.zeros(len(headings), 64, dtype=torch.float32)

    for i, head in enumerate(headings):
        heading_enc[i] = torch.tensor(
                [math.sin(head), math.cos(head)] * (64 // 2))

    return heading_enc.to(device)

def dir_angle_feature(angle_list, device=None):
    feature_dim = 64
    batch_size = len(angle_list)
    max_leng = max([len(k) for k in angle_list]) + 1  # +1 for stop
    heading_enc = torch.zeros(
        batch_size, max_leng, feature_dim, dtype=torch.float32)

    for i in range(batch_size):
        for j, angle_rad in enumerate(angle_list[i]):
            heading_enc[i][j] = torch.tensor(
                [math.sin(angle_rad), 
                math.cos(angle_rad)] * (feature_dim // 2))

    return heading_enc


def angle_feature_with_ele(headings, device=None):
    # twopi = math.pi * 2
    # heading = (heading + twopi) % twopi     # From 0 ~ 2pi
    # It will be the same
    heading_enc = torch.zeros(len(headings), 128, dtype=torch.float32)

    for i, head in enumerate(headings):
        heading_enc[i] = torch.tensor(
            [
                math.sin(head), math.cos(head),
                math.sin(0.0), math.cos(0.0),  # elevation
            ] * (128 // 4))

    return heading_enc.to(device)

def angle_feature_torch(headings: torch.Tensor):
    return torch.stack(
        [
            torch.sin(headings),
            torch.cos(headings),
            torch.sin(torch.zeros_like(headings)),
            torch.cos(torch.zeros_like(headings))
        ]
    ).float().T

def dir_angle_feature_with_ele(angle_list, device=None):
    feature_dim = 128
    batch_size = len(angle_list)
    max_leng = max([len(k) for k in angle_list]) + 1  # +1 for stop
    heading_enc = torch.zeros(
        batch_size, max_leng, feature_dim, dtype=torch.float32)

    for i in range(batch_size):
        for j, angle_rad in enumerate(angle_list[i]):
            heading_enc[i][j] = torch.tensor(
            [
                math.sin(angle_rad), math.cos(angle_rad),
                math.sin(0.0), math.cos(0.0),  # elevation
            ] * (128 // 4))

    return heading_enc
    

def length2mask(length, size=None):
    batch_size = len(length)
    size = int(max(length)) if size is None else size
    mask = (torch.arange(size, dtype=torch.int64).unsqueeze(0).repeat(batch_size, 1)
                > (torch.LongTensor(length) - 1).unsqueeze(1)).cuda()
    return mask