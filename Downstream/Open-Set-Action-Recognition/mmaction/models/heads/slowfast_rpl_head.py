import torch
import torch.nn as nn
from mmcv.cnn import normal_init

from ..registry import HEADS
from .base import BaseHead


@HEADS.register_module()
class SlowFastRPLHead(BaseHead):
    """The classification head for SlowFast.

    Args:
        num_classes (int): Number of classes to be classified.
        in_channels (int): Number of channels in input feature.
        loss_cls (dict): Config for building loss.
            Default: dict(type='CrossEntropyLoss').
        spatial_type (str): Pooling type in spatial dimension. Default: 'avg'.
        dropout_ratio (float): Probability of dropout layer. Default: 0.8.
        init_std (float): Std value for Initiation. Default: 0.01.
        kwargs (dict, optional): Any keyword argument to be used to initialize
            the head.
    """

    def __init__(self,
                 num_classes,
                 in_channels,
                 loss_cls=dict(type='RPLoss'),
                 spatial_type='avg',
                 dropout_ratio=0.8,
                 init_std=0.01,
                 num_centers=1,
                 **kwargs):

        super().__init__(num_classes, in_channels, loss_cls, **kwargs)
        self.spatial_type = spatial_type
        self.dropout_ratio = dropout_ratio
        self.init_std = init_std
        self.num_centers = num_centers

        if self.dropout_ratio != 0:
            self.dropout = nn.Dropout(p=self.dropout_ratio)
        else:
            self.dropout = None
        self.fc_centers = nn.Linear(self.in_channels, self.num_classes * self.num_centers, bias=False)

        if self.spatial_type == 'avg':
            self.avg_pool = nn.AdaptiveAvgPool3d((1, 1, 1))
        else:
            self.avg_pool = None

    def init_weights(self):
        """Initiate the parameters from scratch."""
        normal_init(self.fc_centers, std=self.init_std)

    def compute_dist(self, features, center=None, metric='fc'):
        if metric == 'l2':
            f_2 = torch.sum(torch.pow(features, 2), dim=1, keepdim=True)
            if center is None:
                c_2 = torch.sum(torch.pow(self.fc_centers.weight, 2), dim=1, keepdim=True)
                dist = f_2 - 2 * self.fc_centers(features) + torch.transpose(c_2, 1, 0)
            else:
                c_2 = torch.sum(torch.pow(center, 2), dim=1, keepdim=True)
                dist = f_2 - 2*torch.matmul(features, torch.transpose(center, 1, 0)) + torch.transpose(c_2, 1, 0)
            dist = dist / float(features.shape[1])
        else:
            if center is None:
                dist = self.fc_centers(features)
            else:
                dist = features.matmul(center.t())
        dist = torch.reshape(dist, [-1, self.num_classes, self.num_centers])
        dist = torch.mean(dist, dim=2)
        return dist

    def forward(self, x):
        """Defines the computation performed at every call.

        Args:
            x (torch.Tensor): The input data.

        Returns:
            torch.Tensor: The classification scores for input samples.
        """
        # ([N, channel_fast, T, H, W], [(N, channel_slow, T, H, W)])
        x_fast, x_slow = x
        # ([N, channel_fast, 1, 1, 1], [N, channel_slow, 1, 1, 1])
        x_fast = self.avg_pool(x_fast)
        x_slow = self.avg_pool(x_slow)
        # [N, channel_fast + channel_slow, 1, 1, 1]
        x = torch.cat((x_slow, x_fast), dim=1)

        if self.dropout is not None:
            x = self.dropout(x)

        # [N x C]
        x = x.view(x.size(0), -1)
        # [N x num_classes]
        dist = self.compute_dist(x)
        # [N, num_classes]
        if self.loss_cls.__class__.__name__ == 'GCPLoss':
            dist = -dist
        outputs = {'dist': dist, 'feature': x, 'centers': self.fc_centers.weight}

        return outputs
