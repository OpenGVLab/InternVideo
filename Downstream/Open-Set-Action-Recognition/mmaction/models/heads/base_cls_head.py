# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.nn as nn
from mmcv.cnn import normal_init

from ..builder import HEADS
from .base import BaseHead
import pdb

@HEADS.register_module()
class BaseClsHead(BaseHead):
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
                 loss_cls=dict(type='CrossEntropyLoss'),
                 dropout_ratio=0.5,
                 init_std=0.01,
                 **kwargs):

        super().__init__(num_classes, in_channels, loss_cls, **kwargs)
        self.init_std = init_std
        self.dropout_ratio = dropout_ratio
        self.fc_cls = nn.Linear(in_channels, num_classes)

        if self.dropout_ratio != 0:
            self.dropout = nn.Dropout(p=self.dropout_ratio)
        else:
            self.dropout = None

    def init_weights(self):
        """Initiate the parameters from scratch."""
        normal_init(self.fc_cls, std=self.init_std)

    def forward(self, x):
        """Defines the computation performed at every call.

        Args:
            x (torch.Tensor): The input data.

        Returns:
            torch.Tensor: The classification scores for input samples.
        """
        if self.dropout is not None:
            x = self.dropout(x)

        # [N x num_classes]
        cls_score = self.fc_cls(x)

        return cls_score
