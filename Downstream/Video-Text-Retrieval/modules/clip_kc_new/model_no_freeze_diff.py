import os
import time

import torch
import torch.nn as nn
from fvcore.nn import FlopCountAnalysis
from fvcore.nn import flop_count_table

import evl_utils
from evl_utils import TransformerDecoder_uniformer_diff_conv_balance

PATH_PREFIX = '/mnt/lustre/share_data/likunchang.vendor/code/EVL/clip_kc/model'


class EVL(nn.Module):
    def __init__(self, 
        backbone='vit_b16',
        n_layers=12, 
        n_dim=1024, 
        n_head=16, 
        mlp_factor=4.0, 
        drop_path_rate=0.,
        mlp_dropout=[0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5], 
        cls_dropout=0.5, 
        num_frames=8,
        t_size=8, 
        use_t_conv=True,
        use_t_pos_embed=True,
        uni_layer=4,
        uni_type='3d',
        add_ffn=True,
        t_conv_type='1d',
        pre_prompt=True,
        balance=0.,
        after_me=True, 
        before_me=False, 
        me_type='dstm', 
        me_reduction=4,
        num_classes=400,
    ):
        super().__init__()

        # pre-trained from CLIP
        self.backbone = evl_utils.__dict__[backbone](pretrained=False, num_frames=num_frames, t_size=t_size)

        self.evl = TransformerDecoder_uniformer_diff_conv_balance(
            n_layers=n_layers, n_dim=n_dim, n_head=n_head, 
            mlp_factor=mlp_factor, drop_path_rate=drop_path_rate,
            mlp_dropout=mlp_dropout, cls_dropout=cls_dropout, t_size=t_size, 
            use_t_conv=use_t_conv, use_t_pos_embed=use_t_pos_embed,
            uni_layer=uni_layer, uni_type=uni_type, add_ffn=add_ffn, t_conv_type=t_conv_type, 
            pre_prompt=pre_prompt, balance=balance,
            after_me=after_me, before_me=before_me, 
            me_type=me_type, me_reduction=me_reduction,
            num_classes=num_classes
        )
        self.return_num = n_layers

    def forward(self, x, mode='image'):
        features = self.backbone(x, return_num=self.return_num, mode=mode)
        output = self.evl(features, mode=mode)

        return output


def cal_flops(model, frame=8, size=224):
    flops = FlopCountAnalysis(model, torch.rand(1, 3, frame, size, size))
    s = time.time()
    print(flop_count_table(flops, max_depth=1))
    print(time.time()-s)


def vit_2plus1d_diff_b_sparse8(pretrained=True):
    # 8x224x224
    # k400 1x1: 82.5
    model = EVL(
        backbone='vit_2plus1d_dw_bias_b16',
        n_layers=12, 
        n_dim=768, 
        n_head=12, 
        mlp_factor=4.0, 
        drop_path_rate=0.,
        mlp_dropout=[0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5], 
        cls_dropout=0.5, 
        num_frames=8,
        t_size=8, 
        use_t_conv=True,
        use_t_pos_embed=True,
        uni_layer=0,
        uni_type='2d',
        add_ffn=False,
        t_conv_type='3d',
        pre_prompt=False,
        balance=0.,
        after_me=True, 
        before_me=False, 
        me_type='stm', 
        me_reduction=4,
        num_classes=400,
    )
    # if pretrained:
    #     pretrained_path = os.path.join(PATH_PREFIX, 'vit_2plus1d_diff_b_sparse8.pyth')
    #     print(f'lodel model from: {pretrained_path}')
    #     state_dict = torch.load(pretrained_path, map_location='cpu')
    #     model.load_state_dict(state_dict)
    return model


if __name__ == '__main__':

    model = vit_2plus1d_diff_b_sparse8()
    cal_flops(model, frame=1, size=224)