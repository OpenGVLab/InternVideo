from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import torch
import torch.nn as nn
from alphaction.layers import FrozenBatchNorm3d


class NLBlock(nn.Module):
    def __init__(self, dim_in, dim_out, dim_inner, nl_cfg, group=False):
        super(NLBlock, self).__init__()

        self.nl_cfg = nl_cfg.clone()
        self.group = group
        self.group_size = 4

        init_std = nl_cfg.CONV_INIT_STD
        bias = not nl_cfg.NO_BIAS
        pool_stride = 2

        self.scale_value = dim_inner ** (-0.5)
        self.dim_inner = dim_inner

        self.theta = nn.Conv3d(dim_in, dim_inner, 1, bias=bias)
        nn.init.normal_(self.theta.weight, std=init_std)
        if bias:
            nn.init.constant_(self.theta.bias, 0)

        if nl_cfg.USE_MAXPOOL:
            self.maxpool = nn.MaxPool3d((1, pool_stride, pool_stride),
                                        stride=(1, pool_stride, pool_stride))

        self.phi = nn.Conv3d(dim_in, dim_inner, 1, bias=bias)
        nn.init.normal_(self.phi.weight, std=init_std)
        if bias:
            nn.init.constant_(self.phi.bias, 0)

        self.g = nn.Conv3d(dim_in, dim_inner, 1, bias=bias)
        nn.init.normal_(self.g.weight, std=init_std)
        if bias:
            nn.init.constant_(self.g.bias, 0)

        if nl_cfg.USE_SOFTMAX:
            self.softmax = nn.Softmax(dim=2)

        self.out = nn.Conv3d(dim_inner, dim_out, 1, bias=bias)
        if nl_cfg.USE_ZERO_INIT_CONV:
            nn.init.constant_(self.out.weight, 0)
        else:
            nn.init.normal_(self.out.weight, std=init_std)
        if bias:
            nn.init.constant_(self.out.bias, 0)

        if nl_cfg.USE_BN:
            if nl_cfg.FROZEN_BN:
                self.bn = FrozenBatchNorm3d(dim_out, eps=nl_cfg.BN_EPSILON)
            else:
                self.bn = nn.BatchNorm3d(dim_out, eps=nl_cfg.BN_EPSILON, momentum=nl_cfg.BN_MOMENTUM)
            nn.init.constant_(self.bn.weight, nl_cfg.BN_INIT_GAMMA)

    def forward(self, x):
        if x.dim() != 5:
            raise ValueError('expected 4D or 5D input (got {}D input)'
                             .format(x.dim()))

        if self.group:
            x = x.transpose(1, 2)
            sz_before_group = list(x.shape)
            sz_after_group = sz_before_group.copy()
            sz_after_group[0] = -1
            sz_after_group[1] = self.group_size
            x = x.contiguous().view(*sz_after_group)
            x = x.transpose(1, 2)

        batch_size = x.shape[0]

        theta = self.theta(x)

        if self.nl_cfg.USE_MAXPOOL:
            max_pool = self.maxpool(x)
        else:
            max_pool = x

        phi = self.phi(max_pool)

        g = self.g(max_pool)

        org_size = theta.size()
        mat_size = [batch_size, self.dim_inner, -1]
        theta = theta.view(*mat_size)
        phi = phi.view(*mat_size)
        g = g.view(*mat_size)

        theta_phi = torch.bmm(theta.transpose(1, 2), phi)

        if self.nl_cfg.USE_SOFTMAX:
            if self.nl_cfg.USE_SCALE:
                theta_phi_sc = theta_phi * self.scale_value
            else:
                theta_phi_sc = theta_phi
            p = self.softmax(theta_phi_sc)
        else:
            p = theta_phi / theta_phi.shape[-1]

        t = torch.bmm(g, p.transpose(1, 2))

        t = t.view(org_size)

        out = self.out(t)

        if self.nl_cfg.USE_BN:
            out = self.bn(out)
        out = out + x

        if self.group:
            out = out.transpose(1, 2)
            out = out.contiguous().view(*sz_before_group)
            out = out.transpose(1, 2)

        return out

    def c2_weight_mapping(self):
        weight_map = {}
        for name, m_child in self.named_children():
            if m_child.state_dict():
                if isinstance(m_child, (nn.BatchNorm3d, FrozenBatchNorm3d)):
                    weight_map[name + '.weight'] = '{}_s'.format(name)
                    weight_map[name + '.running_mean'] = '{}_rm'.format(name)
                    weight_map[name + '.running_var'] = '{}_riv'.format(name)
                elif isinstance(m_child, nn.GroupNorm):
                    weight_map[name + '.weight'] = '{}_s'.format(name)
                else:
                    weight_map[name + '.weight'] = '{}_w'.format(name)
                weight_map[name + '.bias'] = '{}_b'.format(name)
        return weight_map
