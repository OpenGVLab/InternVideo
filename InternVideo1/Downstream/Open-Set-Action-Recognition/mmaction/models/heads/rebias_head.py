import torch
import torch.nn as nn
from mmcv.cnn import ConvModule, constant_init, normal_init, xavier_init
from ..builder import build_loss


class RebiasHead(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 loss_weight=0.5,
                 loss_rebias=dict(type='RebiasLoss')):
        super().__init__()

        self.conv_f = ConvModule(
            in_channels,
            in_channels * 2, (1, 3, 3),
            stride=(1, 2, 2),
            padding=(0, 1, 1),
            bias=False,
            conv_cfg=dict(type='Conv3d'),
            norm_cfg=dict(type='BN3d', requires_grad=True))
        self.conv_g = ConvModule(
            in_channels,
            in_channels * 2, (1, 3, 3),
            stride=(1, 2, 2),
            padding=(0, 1, 1),
            bias=False,
            conv_cfg=dict(type='Conv3d'),
            norm_cfg=dict(type='BN3d', requires_grad=True))
        self.avg_pool = nn.AdaptiveAvgPool3d((1, 1, 1))
        self.loss_weight = loss_weight
        self.dropout = nn.Dropout(p=0.5)
        self.fc_f = nn.Linear(in_channels * 2, out_channels)
        self.fc_g = nn.Linear(in_channels * 2, out_channels)
        self.loss_rebias = build_loss(loss_rebias)

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                normal_init(m, std=0.01)
            if isinstance(m, nn.Conv3d):
                xavier_init(m, distribution='uniform')
            if isinstance(m, nn.BatchNorm3d):
                constant_init(m, 1)

    def forward(self, x, target=None):
        # x: (B, 1024, 8, 14, 14)
        xs = x.detach().clone()  # we do not want the backbone is updated by g(xs)
        xs = xs[:, :, torch.randperm(xs.size()[2])]  # temporally shuffle the feature
        losses = dict()
        if target is None:
            return losses
        # f(x)
        x = self.conv_f(x)  # (B, 2048, 8, 7, 7)
        x = self.avg_pool(x).squeeze(-1).squeeze(-1).squeeze(-1)
        # phi(f(x))
        y = self.dropout(x)
        y = self.fc_f(y)

        # g(xs)
        xs = self.conv_g(xs)  # (B, 2048, 8, 7, 7)
        xs = self.avg_pool(xs).squeeze(-1).squeeze(-1).squeeze(-1)
        # phi(g(xs))
        ys = self.dropout(xs)
        ys = self.fc_g(ys)

        if target.shape == torch.Size([]):
            target = target.unsqueeze(0)

        # compute the rebias losses
        rebias_loss = self.loss_rebias(x, xs, y, ys, target)

        if isinstance(rebias_loss, dict):
            for k, v in rebias_loss.items():
                losses.update({k: self.loss_weight * v})
        else:
            losses = {'loss_rebias': rebias_loss}
        return losses