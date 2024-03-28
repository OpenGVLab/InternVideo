import torch
import torch.nn as nn
from mmcv.cnn import normal_init

from ..registry import HEADS
from .tsn_head import TSNHead


@HEADS.register_module()
class TPNRPLHead(TSNHead):
    """Class head for TPN.

    Args:
        num_classes (int): Number of classes to be classified.
        in_channels (int): Number of channels in input feature.
        loss_cls (dict): Config for building loss.
            Default: dict(type='CrossEntropyLoss').
        spatial_type (str): Pooling type in spatial dimension. Default: 'avg'.
        consensus (dict): Consensus config dict.
        dropout_ratio (float): Probability of dropout layer. Default: 0.4.
        init_std (float): Std value for Initiation. Default: 0.01.
        multi_class (bool): Determines whether it is a multi-class
            recognition task. Default: False.
        label_smooth_eps (float): Epsilon used in label smooth.
            Reference: https://arxiv.org/abs/1906.02629. Default: 0.
    """

    def __init__(self, num_centers=1, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.num_centers = num_centers
        self.fc_centers = nn.Linear(self.in_channels, self.num_classes * self.num_centers, bias=False)

        if self.spatial_type == 'avg':
            # use `nn.AdaptiveAvgPool3d` to adaptively match the in_channels.
            self.avg_pool3d = nn.AdaptiveAvgPool3d((1, 1, 1))
        else:
            self.avg_pool3d = None

        self.avg_pool2d = None

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


    def forward(self, x, num_segs=None):
        """Defines the computation performed at every call.

        Args:
            x (torch.Tensor): The input data.
            num_segs (int | None): Number of segments into which a video
                is divided. Default: None.
        Returns:
            torch.Tensor: The classification scores for input samples.
        """
        if self.avg_pool2d is None:
            kernel_size = (1, x.shape[-2], x.shape[-1])
            self.avg_pool2d = nn.AvgPool3d(kernel_size, stride=1, padding=0)

        if num_segs is None:
            # [N, in_channels, 3, 7, 7]
            x = self.avg_pool3d(x)
        else:
            # [N * num_segs, in_channels, 7, 7]
            x = self.avg_pool2d(x)
            # [N * num_segs, in_channels, 1, 1]
            x = x.reshape((-1, num_segs) + x.shape[1:])
            # [N, num_segs, in_channels, 1, 1]
            x = self.consensus(x)
            # [N, 1, in_channels, 1, 1]
            x = x.squeeze(1)
            # [N, in_channels, 1, 1]
        if self.dropout is not None:
            x = self.dropout(x)
            # [N, in_channels, 1, 1]
        x = x.view(x.size(0), -1)
        # [N, in_channels]
        dist = self.compute_dist(x)
        # [N, num_classes]
        if self.loss_cls.__class__.__name__ == 'GCPLoss':
            dist = -dist
        outputs = {'dist': dist, 'feature': x, 'centers': self.fc_centers.weight}

        return outputs
