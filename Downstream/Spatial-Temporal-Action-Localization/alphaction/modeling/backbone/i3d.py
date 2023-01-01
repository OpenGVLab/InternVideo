from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import torch.nn as nn
from alphaction.layers import FrozenBatchNorm3d
from alphaction.modeling.common_blocks import ResNLBlock


def get_model_cfg(cfg):
    backbone_strs = cfg.MODEL.BACKBONE.CONV_BODY.split('-')[1:]
    error_msg = 'Model backbone {} is not supported.'.format(cfg.MODEL.BACKBONE.CONV_BODY)

    use_temp_convs_1 = [2]
    temp_strides_1 = [2]
    max_pool_stride_1 = 2

    use_temp_convs_2 = [1, 1, 1]
    temp_strides_2 = [1, 1, 1]
    max_pool_stride_2 = 2

    use_temp_convs_3 = [1, 0, 1, 0]
    temp_strides_3 = [1, 1, 1, 1]

    use_temp_convs_5 = [0, 1, 0]
    temp_strides_5 = [1, 1, 1]

    avg_pool_stride = int(cfg.INPUT.FRAME_NUM / 8)
    if backbone_strs[0] == 'Resnet50':
        block_config = (3, 4, 6, 3)

        use_temp_convs_4 = [1, 0, 1, 0, 1, 0]
        temp_strides_4 = [1, 1, 1, 1, 1, 1]
    elif backbone_strs[0] == 'Resnet101':
        block_config = (3, 4, 23, 3)

        use_temp_convs_4 = []
        for i in range(23):
            if i % 2 == 0:
                use_temp_convs_4.append(1)
            else:
                use_temp_convs_4.append(0)
        temp_strides_4 = [1, ] * 23
    else:
        raise KeyError(error_msg)

    if len(backbone_strs) > 1:
        if len(backbone_strs) == 2 and backbone_strs[1] == 'Sparse':
            temp_strides_1 = [1]
            max_pool_stride_1 = 1
            avg_pool_stride = int(cfg.INPUT.FRAME_NUM / 2)
        else:
            raise KeyError(error_msg)

    use_temp_convs_set = [use_temp_convs_1, use_temp_convs_2, use_temp_convs_3, use_temp_convs_4, use_temp_convs_5]
    temp_strides_set = [temp_strides_1, temp_strides_2, temp_strides_3, temp_strides_4, temp_strides_5]
    pool_strides_set = [max_pool_stride_1, max_pool_stride_2, avg_pool_stride]
    return block_config, use_temp_convs_set, temp_strides_set, pool_strides_set


class I3D(nn.Module):
    def __init__(self, cfg):
        super(I3D, self).__init__()

        self.cfg = cfg.clone()

        block_config, use_temp_convs_set, temp_strides_set, pool_strides_set = get_model_cfg(cfg)
        conv3_nonlocal = cfg.MODEL.BACKBONE.I3D.CONV3_NONLOCAL
        conv4_nonlocal = cfg.MODEL.BACKBONE.I3D.CONV4_NONLOCAL

        dim_inner = 64
        conv_dims = [64, 256, 512, 1024, 2048]
        self.dim_out = conv_dims[-1]
        n1, n2, n3, n4 = block_config
        layer_mod = 2
        conv3_nl_mod = layer_mod
        conv4_nl_mod = layer_mod
        if not conv3_nonlocal:
            conv3_nl_mod = 1000
        if not conv4_nonlocal:
            conv4_nl_mod = 1000
        self.c2_mapping = None

        data_dim = 3
        self.conv1 = nn.Conv3d(data_dim, conv_dims[0], (1 + use_temp_convs_set[0][0] * 2, 7, 7),
                               stride=(temp_strides_set[0][0], 2, 2),
                               padding=(use_temp_convs_set[0][0], 3, 3), bias=False)
        nn.init.kaiming_normal_(self.conv1.weight)

        if cfg.MODEL.BACKBONE.FROZEN_BN:
            self.bn1 = FrozenBatchNorm3d(conv_dims[0], eps=cfg.MODEL.BACKBONE.BN_EPSILON)
            nn.init.constant_(self.bn1.weight, 1.0)
            nn.init.constant_(self.bn1.bias, 0.0)
        else:
            self.bn1 = nn.BatchNorm3d(conv_dims[0], eps=cfg.MODEL.BACKBONE.BN_EPSILON, momentum=cfg.MODEL.BACKBONE.BN_MOMENTUM)

        self.relu = nn.ReLU(inplace=True)
        self.maxpool1 = nn.MaxPool3d((pool_strides_set[0], 3, 3), stride=(pool_strides_set[0], 2, 2))

        self.res_nl1 = ResNLBlock(cfg, conv_dims[0], conv_dims[1], stride=1, num_blocks=n1, dim_inner=dim_inner,
                                  use_temp_convs=use_temp_convs_set[1], temp_strides=temp_strides_set[1])
        self.maxpool2 = nn.MaxPool3d((pool_strides_set[1], 1, 1), stride=(pool_strides_set[1], 1, 1))

        self.res_nl2 = ResNLBlock(cfg, conv_dims[1], conv_dims[2], stride=2, num_blocks=n2,
                                  dim_inner=dim_inner * 2, use_temp_convs=use_temp_convs_set[2],
                                  temp_strides=temp_strides_set[2], nonlocal_mod=conv3_nl_mod,
                                  group_nonlocal=cfg.MODEL.BACKBONE.I3D.CONV3_GROUP_NL)

        self.res_nl3 = ResNLBlock(cfg, conv_dims[2], conv_dims[3], stride=2, num_blocks=n3,
                                  dim_inner=dim_inner * 4, use_temp_convs=use_temp_convs_set[3],
                                  temp_strides=temp_strides_set[3], nonlocal_mod=conv4_nl_mod)

        self.res_nl4 = ResNLBlock(cfg, conv_dims[3], conv_dims[4], stride=1, num_blocks=n4,
                                  dim_inner=dim_inner * 8, use_temp_convs=use_temp_convs_set[4],
                                  temp_strides=temp_strides_set[4],
                                  dilation=2)

    def forward(self, _, x):
        # We only use fast videos, which is the second input.
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.maxpool1(out)

        out = self.res_nl1(out)
        out = self.maxpool2(out)

        out = self.res_nl2(out)

        out = self.res_nl3(out)

        out = self.res_nl4(out)
        return None, out

    def c2_weight_mapping(self):
        if self.c2_mapping is None:
            weight_map = {'conv1.weight': 'conv1_w',
                      'bn1.weight': 'res_conv1_bn_s',
                      'bn1.bias': 'res_conv1_bn_b',
                      'bn1.running_mean': 'res_conv1_bn_rm',
                      'bn1.running_var': 'res_conv1_bn_riv'}
            for i in range(1, 5):
                name = 'res_nl{}'.format(i)
                child_map = getattr(self, name).c2_weight_mapping()
                for key, val in child_map.items():
                    new_key = name + '.' + key
                    weight_map[new_key] = val.format(i + 1)
            self.c2_mapping = weight_map
        return self.c2_mapping
