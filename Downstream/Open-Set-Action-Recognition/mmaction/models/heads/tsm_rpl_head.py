import torch
import torch.nn as nn
from mmcv.cnn import normal_init

from ..registry import HEADS
from .base import AvgConsensus, BaseHead


@HEADS.register_module()
class TSMRPLHead(BaseHead):
    """Class head for TSM.

    Args:
        num_classes (int): Number of classes to be classified.
        in_channels (int): Number of channels in input feature.
        num_segments (int): Number of frame segments. Default: 8.
        loss_cls (dict): Config for building loss.
            Default: dict(type='CrossEntropyLoss')
        spatial_type (str): Pooling type in spatial dimension. Default: 'avg'.
        consensus (dict): Consensus config dict.
        dropout_ratio (float): Probability of dropout layer. Default: 0.4.
        init_std (float): Std value for Initiation. Default: 0.01.
        is_shift (bool): Indicating whether the feature is shifted.
            Default: True.
        temporal_pool (bool): Indicating whether feature is temporal pooled.
            Default: False.
        kwargs (dict, optional): Any keyword argument to be used to initialize
            the head.
    """

    def __init__(self,
                 num_classes,
                 in_channels,
                 num_segments=8,
                 loss_cls=dict(type='RPLoss'),
                 spatial_type='avg',
                 consensus=dict(type='AvgConsensus', dim=1),
                 dropout_ratio=0.8,
                 init_std=0.001,
                 is_shift=True,
                 temporal_pool=False,
                 num_centers=1,
                 **kwargs):
        super().__init__(num_classes, in_channels, loss_cls, **kwargs)

        self.spatial_type = spatial_type
        self.dropout_ratio = dropout_ratio
        self.num_segments = num_segments
        self.init_std = init_std
        self.is_shift = is_shift
        self.temporal_pool = temporal_pool
        self.num_centers = num_centers

        consensus_ = consensus.copy()

        consensus_type = consensus_.pop('type')
        if consensus_type == 'AvgConsensus':
            self.consensus = AvgConsensus(**consensus_)
        else:
            self.consensus = None

        if self.dropout_ratio != 0:
            self.dropout = nn.Dropout(p=self.dropout_ratio)
        else:
            self.dropout = None
        self.fc_centers = nn.Linear(self.in_channels, self.num_classes * self.num_centers, bias=False)

        if self.spatial_type == 'avg':
            # use `nn.AdaptiveAvgPool2d` to adaptively match the in_channels.
            self.avg_pool = nn.AdaptiveAvgPool2d(1)
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


    def aggregate(self, cls_score):
        if self.is_shift and self.temporal_pool:
            # [2 * N, num_segs // 2, num_classes]
            cls_score = cls_score.view((-1, self.num_segments // 2) +
                                       cls_score.size()[1:])
        else:
            # [N, num_segs, num_classes]
            cls_score = cls_score.view((-1, self.num_segments) +
                                       cls_score.size()[1:])
        # [N, 1, num_classes]
        cls_score = self.consensus(cls_score)
        # [N, num_classes]
        return cls_score.squeeze(1)


    def forward(self, x, num_segs):
        """Defines the computation performed at every call.

        Args:
            x (torch.Tensor): The input data.
            num_segs (int): Useless in TSMHead. By default, `num_segs`
                is equal to `clip_len * num_clips * num_crops`, which is
                automatically generated in Recognizer forward phase and
                useless in TSM models. The `self.num_segments` we need is a
                hyper parameter to build TSM models.
        Returns:
            torch.Tensor: The classification scores for input samples.
        """
        # [N * num_segs, in_channels, 7, 7]
        x = self.avg_pool(x)
        # [N * num_segs, in_channels, 1, 1]
        x = torch.flatten(x, 1)
        # [N * num_segs, in_channels]
        if self.dropout is not None:
            x = self.dropout(x)
        # [N * num_segs, num_classes]
        dist = self.compute_dist(x)
        # [N, num_classes]

        dist = self.aggregate(dist)
        feature = self.aggregate(x)

        if self.loss_cls.__class__.__name__ == 'GCPLoss':
            dist = -dist
        outputs = {'dist': dist, 'feature': feature, 'centers': self.fc_centers.weight}

        return outputs
