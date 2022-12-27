import os
import time

import torch
import torch.nn as nn
from fvcore.nn import FlopCountAnalysis
from fvcore.nn import flop_count_table

import evl_utils
from evl_utils import TransformerDecoder

PATH_PREFIX = '/mnt/lustre/share_data/likunchang.vendor/code/EVL/clip_kc/model'


class EVL(nn.Module):
    def __init__(self, 
        backbone='vit_l14_336',
        n_layers=4, 
        n_dim=1024, 
        n_head=16, 
        mlp_factor=4.0, 
        drop_path_rate=0.,
        mlp_dropout=[0.5, 0.5, 0.5, 0.5], 
        cls_dropout=0.5, 
        t_size=32, 
        use_t_conv=True,
        use_t_pos_embed=True,
        num_classes=400
    ):
        super().__init__()

        # pre-trained from CLIP
        self.backbone = evl_utils.__dict__[backbone](pretrained=False)

        self.evl = TransformerDecoder(
            n_layers=n_layers, n_dim=n_dim, n_head=n_head, 
            mlp_factor=mlp_factor, drop_path_rate=drop_path_rate,
            mlp_dropout=mlp_dropout, cls_dropout=cls_dropout, t_size=t_size, 
            use_t_conv=use_t_conv, use_t_pos_embed=use_t_pos_embed,
            num_classes=num_classes,
        )
        self.return_num = n_layers

    def forward(self, x):
        features = self.backbone(x, return_num=self.return_num)
        output = self.evl(features)

        return output


def cal_flops(model, frame=8, size=224):
    flops = FlopCountAnalysis(model, torch.rand(1, 3, frame, size, size))
    s = time.time()
    print(flop_count_table(flops, max_depth=1))
    print(time.time()-s)


def vit_l_sparse16(pretrained=True):
    # 16x224x224
    # k400 1x1: 86.5
    model = EVL(
        backbone='vit_l14',
        n_layers=4, 
        n_dim=1024, 
        n_head=16, 
        mlp_factor=4.0, 
        drop_path_rate=0.,
        mlp_dropout=[0.5, 0.5, 0.5, 0.5], 
        cls_dropout=0.5, 
        t_size=32, 
        use_t_conv=True,
        use_t_pos_embed=True,
        num_classes=400
    )
    if pretrained:
        pretrained_path = os.path.join(PATH_PREFIX, 'vit_l_sparse16.pyth')
        print(f'lodel model from: {pretrained_path}')
        state_dict = torch.load(pretrained_path, map_location='cpu')
        model.load_state_dict(state_dict)
    return model


def vit_l_sparse32(pretrained=True):
    # 32x224x224
    # k400 1x1: 87.0
    model = EVL(
        backbone='vit_l14',
        n_layers=4, 
        n_dim=1024, 
        n_head=16, 
        mlp_factor=4.0, 
        drop_path_rate=0.,
        mlp_dropout=[0.5, 0.5, 0.5, 0.5], 
        cls_dropout=0.5, 
        t_size=32, 
        use_t_conv=True,
        use_t_pos_embed=True,
        num_classes=400
    )
    if pretrained:
        pretrained_path = os.path.join(PATH_PREFIX, 'vit_l_sparse32.pyth')
        print(f'lodel model from: {pretrained_path}')
        state_dict = torch.load(pretrained_path, map_location='cpu')
        model.load_state_dict(state_dict)
    return model


def vit_l336_sparse32(pretrained=True):
    # 32x336x336
    # k400 1x1: 87.4
    model = EVL(
        backbone='vit_l14_336',
        n_layers=4, 
        n_dim=1024, 
        n_head=16, 
        mlp_factor=4.0, 
        drop_path_rate=0.,
        mlp_dropout=[0.5, 0.5, 0.5, 0.5], 
        cls_dropout=0.5, 
        t_size=32, 
        use_t_conv=True,
        use_t_pos_embed=True,
        num_classes=400
    )
    if pretrained:
        pretrained_path = os.path.join(PATH_PREFIX, 'vit_l336_sparse32.pyth')
        print(f'lodel model from: {pretrained_path}')
        state_dict = torch.load(pretrained_path, map_location='cpu')
        model.load_state_dict(state_dict)
    return model


if __name__ == '__main__':

    model = vit_l_sparse16()
    cal_flops(model, frame=16, size=224)

    # model = vit_l_sparse32()
    # cal_flops(model, frame=32, size=224)

    # model = vit_l336_sparse32()
    # cal_flops(model, frame=32, size=336)