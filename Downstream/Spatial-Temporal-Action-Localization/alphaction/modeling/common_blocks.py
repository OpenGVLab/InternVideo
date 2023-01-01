import torch.nn as nn
from alphaction.modeling.nonlocal_block import NLBlock
from alphaction.layers import FrozenBatchNorm3d


class Conv3dBN(nn.Module):
    def __init__(self, cfg, dim_in, dim_out, kernels, stride, padding, dilation=1, init_weight=None):
        super(Conv3dBN, self).__init__()
        self.conv = nn.Conv3d(dim_in, dim_out, kernels, stride=stride,
                              padding=padding, dilation=dilation, bias=False)
        nn.init.kaiming_normal_(self.conv.weight)
        if cfg.MODEL.BACKBONE.FROZEN_BN:
            self.bn = FrozenBatchNorm3d(dim_out, eps=cfg.MODEL.BACKBONE.BN_EPSILON)
            nn.init.constant_(self.bn.weight, 1.0)
            nn.init.constant_(self.bn.bias, 0.0)
        else:
            self.bn = nn.BatchNorm3d(dim_out, eps=cfg.MODEL.BACKBONE.BN_EPSILON, momentum=cfg.MODEL.BACKBONE.BN_MOMENTUM)
            if init_weight is not None:
                nn.init.constant_(self.bn.weight, init_weight)

    def forward(self, x):
        out = self.conv(x)
        out = self.bn(out)
        return out

    def c2_weight_mapping(self):
        return {
            'conv.weight': 'w',
            'bn.weight': 'bn_s',
            'bn.bias': 'bn_b',
            'bn.running_mean': 'bn_rm',
            'bn.running_var': 'bn_riv'
        }


class Bottleneck(nn.Module):
    def __init__(self, cfg, dim_in, dim_out, dim_inner, stride, dilation=1, use_temp_conv=1, temp_stride=1):
        super(Bottleneck, self).__init__()
        self.conv1 = Conv3dBN(cfg, dim_in, dim_inner, (1 + use_temp_conv * 2, 1, 1),
                                 stride=(temp_stride, 1, 1), padding=(use_temp_conv, 0, 0))
        self.conv2 = Conv3dBN(cfg, dim_inner, dim_inner, (1, 3, 3), stride=(1, stride, stride),
                                 dilation=(1, dilation, dilation),
                                 padding=(0, dilation, dilation))
        self.conv3 = Conv3dBN(cfg, dim_inner, dim_out, (1, 1, 1), stride=(1, 1, 1),
                                 padding=0, init_weight=cfg.MODEL.BACKBONE.BN_INIT_GAMMA)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        out = self.conv1(x)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.relu(out)
        out = self.conv3(out)
        return out

    def c2_weight_mapping(self):
        weight_map = {}
        for i in range(1, 4):
            name = 'conv{}'.format(i)
            child_map = getattr(self, name).c2_weight_mapping()
            for key, val in child_map.items():
                new_key = name + '.' + key
                prefix = 'branch2{}_'.format(chr(ord('a') + i - 1))
                weight_map[new_key] = prefix + val
        return weight_map


class ResBlock(nn.Module):
    def __init__(self, cfg, dim_in, dim_out, dim_inner, stride, dilation=1, use_temp_conv=0, temp_stride=1, need_shortcut=False):
        super(ResBlock, self).__init__()

        self.btnk = Bottleneck(cfg, dim_in, dim_out, dim_inner=dim_inner, stride=stride, dilation=dilation,
                               use_temp_conv=use_temp_conv, temp_stride=temp_stride)
        if not need_shortcut:
            self.shortcut = None
        else:
            self.shortcut = Conv3dBN(cfg, dim_in, dim_out, (1, 1, 1),
                                     stride=(temp_stride, stride, stride), padding=0)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        tr = self.btnk(x)
        if self.shortcut is None:
            sc = x
        else:
            sc = self.shortcut(x)
        return self.relu(tr + sc)

    def c2_weight_mapping(self):
        weight_map = {}
        for name, m_child in self.named_children():
            if m_child.state_dict():
                child_map = m_child.c2_weight_mapping()
                for key, val in child_map.items():
                    new_key = name + '.' + key
                    if isinstance(m_child, Conv3dBN):
                        prefix = 'branch1_'
                    else:
                        prefix = ''
                    weight_map[new_key] = prefix + val
        return weight_map


class ResNLBlock(nn.Module):
    def __init__(self, cfg, dim_in, dim_out, stride, num_blocks, dim_inner, use_temp_convs, temp_strides,
                 dilation=1, nonlocal_mod=1000, group_nonlocal=False, lateral=False):
        super(ResNLBlock, self).__init__()
        self.blocks = []
        for idx in range(num_blocks):
            block_name = "res_{}".format(idx)
            block_stride = stride if idx == 0 else 1
            block_dilation = dilation
            dim_in0 = dim_in + int(dim_in * cfg.MODEL.BACKBONE.SLOWFAST.BETA * 2) if lateral and (idx == 0) and (
                    cfg.MODEL.BACKBONE.SLOWFAST.LATERAL != 'ttoc_sum') else dim_in
            # To transfer weight from classification model, we change res5_0 from stride 2 to stride 1,
            # and all res5_x layers from dilation 1 to dilation 2. In pretrain, res5_0 with stride 2 need a shortcut conv.
            # idx==0 and dilation!=1 means that it need a short cut in pretrain stage,
            # so we should keep it since we load weight from a pretrained model.
            # if idx!=0, block_stride will not be larger than 1 in pretrain stage.
            need_shortcut = not (dim_in0==dim_out and temp_strides[idx]==1 and block_stride==1) or \
                             (idx==0 and dilation!=1)
            res_module = ResBlock(cfg, dim_in0, dim_out, dim_inner=dim_inner,
                                  stride=block_stride,
                                  dilation=block_dilation,
                                  use_temp_conv=use_temp_convs[idx],
                                  temp_stride=temp_strides[idx],
                                  need_shortcut=need_shortcut)
            self.add_module(block_name, res_module)
            self.blocks.append(block_name)
            dim_in = dim_out
            if idx % nonlocal_mod == nonlocal_mod - 1:
                nl_block_name = "nonlocal_{}".format(idx)
                nl_module = NLBlock(dim_in, dim_in, int(dim_in / 2),
                                    cfg.MODEL.NONLOCAL, group=group_nonlocal)
                self.add_module(nl_block_name, nl_module)
                self.blocks.append(nl_block_name)

    def forward(self, x):
        for layer_name in self.blocks:
            x = getattr(self, layer_name)(x)
        return x

    def c2_weight_mapping(self):
        weight_map = {}
        for name, m_child in self.named_children():
            idx = name.split('_')[-1]
            if m_child.state_dict():
                child_map = m_child.c2_weight_mapping()
                for key, val in child_map.items():
                    new_key = name + '.' + key
                    if isinstance(m_child, NLBlock):
                        prefix = 'nonlocal_conv{}_' + '{}_'.format(idx)
                    else:
                        prefix = 'res{}_' + '{}_'.format(idx)
                    weight_map[new_key] = prefix + val
        return weight_map
