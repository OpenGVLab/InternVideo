import os
import time

import torch
import torch.nn as nn
from fvcore.nn import FlopCountAnalysis
from fvcore.nn import flop_count_table

from . import evl_utils

PATH_PREFIX = '/mnt/lustre/share_data/likunchang.vendor/code/EVL/clip_kc/model'


class EVL(nn.Module):
    def __init__(self, 
        backbone='vit_b16',
        t_size=16, 
        dw_reduction=1.5, 
        backbone_drop_path_rate=0., 
        return_list=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11],
        n_layers=12, 
        n_dim=768, 
        n_head=12, 
        mlp_factor=4.0, 
        drop_path_rate=0.,
        mlp_dropout=[0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5], 
        cls_dropout=0.5, 
        num_classes=174,
    ):
        super().__init__()

        # pre-trained from CLIP
        self.backbone = evl_utils.__dict__[backbone](
            pretrained=False, 
            t_size=t_size,
            dw_reduction=dw_reduction, 
            backbone_drop_path_rate=backbone_drop_path_rate, 
            return_list=return_list, 
            n_layers=n_layers, 
            n_dim=n_dim, 
            n_head=n_head, 
            mlp_factor=mlp_factor, 
            drop_path_rate=drop_path_rate, 
            mlp_dropout=mlp_dropout, 
            cls_dropout=cls_dropout, 
            num_classes=num_classes,
        )

    def forward(self, x, mode='video'):
        output = self.backbone(x, mode=mode)
        return output


def cal_flops(model, frame=8, size=224):
    flops = FlopCountAnalysis(model, torch.rand(1, 3, frame, size, size))
    s = time.time()
    print(flop_count_table(flops, max_depth=1))
    print(time.time()-s)


def vit_fusion_b_sparse16_k400(pretrained=True):
    model = EVL(
        backbone='vit_fusion_b16',
        t_size=16, 
        dw_reduction=1.5, 
        backbone_drop_path_rate=0., 
        return_list=[8, 9, 10, 11],
        n_layers=4, 
        n_dim=768, 
        n_head=12, 
        mlp_factor=4.0, 
        drop_path_rate=0.,
        mlp_dropout=[0.5, 0.5, 0.5, 0.5],
        cls_dropout=0.5, 
        num_classes=400,
    )
    # if pretrained:
    #     pretrained_path = os.path.join(PATH_PREFIX, 'fuck.pyth')
    #     print(f'lodel model from: {pretrained_path}')
    #     state_dict = torch.load(pretrained_path, map_location='cpu')
    #     model.load_state_dict(state_dict)
    return model


def vit_fusion_b_sparse16_sthsth(pretrained=True):
    model = EVL(
        backbone='vit_fusion_b16',
        t_size=16, 
        dw_reduction=1.5, 
        backbone_drop_path_rate=0., 
        return_list=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11],
        n_layers=12, 
        n_dim=768, 
        n_head=12, 
        mlp_factor=4.0, 
        drop_path_rate=0.,
        mlp_dropout=[0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5],
        cls_dropout=0.5, 
        num_classes=174,
    )
    # if pretrained:
    #     pretrained_path = os.path.join(PATH_PREFIX, 'fuck.pyth')
    #     print(f'lodel model from: {pretrained_path}')
    #     state_dict = torch.load(pretrained_path, map_location='cpu')
    #     model.load_state_dict(state_dict)
    return model


if __name__ == '__main__':

    model = vit_fusion_b_sparse16_k400()
    # cal_flops(model, frame=1, size=224)
    cal_flops(model, frame=16, size=224)