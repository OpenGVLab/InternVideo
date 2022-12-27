import torch
import torch.nn as nn
from mmcv.cnn import ConvModule, constant_init, normal_init, xavier_init
from ..builder import build_loss


class AuxHead(nn.Module):
    """Auxiliary Head.

    This auxiliary head is appended to receive stronger supervision,
    leading to enhanced semantics.

    Args:
        in_channels (int): Channel number of input features.
        out_channels (int): Channel number of output features.
        loss_weight (float): weight of loss for the auxiliary head.
            Default: 0.5.
        loss_cls (dict): loss_cls (dict): Config for building loss.
            Default: ``dict(type='CrossEntropyLoss')``.
    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 loss_weight=0.5,
                 loss_cls=dict(type='CrossEntropyLoss')):
        super().__init__()

        self.conv = ConvModule(
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
        self.fc = nn.Linear(in_channels * 2, out_channels)
        self.loss_cls = build_loss(loss_cls)

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                normal_init(m, std=0.01)
            if isinstance(m, nn.Conv3d):
                xavier_init(m, distribution='uniform')
            if isinstance(m, nn.BatchNorm3d):
                constant_init(m, 1)

    def forward(self, x, target=None):
        losses = dict()
        if target is None:
            return losses
        x = self.conv(x)
        x = self.avg_pool(x).squeeze(-1).squeeze(-1).squeeze(-1)
        x = self.dropout(x)
        x = self.fc(x)

        if target.shape == torch.Size([]):
            target = target.unsqueeze(0)

        losses['loss_aux'] = self.loss_weight * self.loss_cls(x, target)
        return losses