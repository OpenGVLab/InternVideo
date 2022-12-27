import torch.nn as nn
import torch
from mmcv.cnn import ConvModule, constant_init, normal_init, xavier_init
import numpy as np

from ..registry import HEADS
from .base import BaseHead

@HEADS.register_module()
class DebiasHead(BaseHead):
    """Debias head.

    Args:
        num_classes (int): Number of classes to be classified.
        in_channels (int): Number of channels in input feature.
        loss_cls (dict): Config for building loss.
            Default: dict(type='EvidenceLoss')
        spatial_type (str): Pooling type in spatial dimension. Default: 'avg'.
        dropout_ratio (float): Probability of dropout layer. Default: 0.5.
        init_std (float): Std value for Initiation. Default: 0.01.
        kwargs (dict, optional): Any keyword argument to be used to initialize
            the head.
    """

    def __init__(self,
                 num_classes,
                 in_channels,
                 loss_cls=dict(type='EvidenceLoss'),
                 loss_factor=0.1,
                 hsic_factor=0.5,  # useful when alternative=True
                 alternative=False,
                 bias_input=True,
                 bias_network=True,
                 dropout_ratio=0.5,
                 init_std=0.01,
                 **kwargs):
        super().__init__(num_classes, in_channels, loss_cls, **kwargs)
        self.bias_input = bias_input
        self.bias_network = bias_network
        assert bias_input or bias_network, "At least one of the choices (bias_input, bias_network) should be True!"
        self.loss_factor = loss_factor
        self.hsic_factor = hsic_factor
        self.alternative = alternative
        self.f1_conv3d = ConvModule(
            in_channels,
            in_channels * 2, (1, 3, 3),
            stride=(1, 2, 2),
            padding=(0, 1, 1),
            bias=False,
            conv_cfg=dict(type='Conv3d'),
            norm_cfg=dict(type='BN3d', requires_grad=True))
        if bias_input:
            self.f2_conv3d = ConvModule(
                in_channels,
                in_channels * 2, (1, 3, 3),
                stride=(1, 2, 2),
                padding=(0, 1, 1),
                bias=False,
                conv_cfg=dict(type='Conv3d'),
                norm_cfg=dict(type='BN3d', requires_grad=True))
        if bias_network:
            self.f3_conv2d = ConvModule(
                in_channels,
                in_channels * 2, (3, 3),
                stride=(2, 2),
                padding=(1, 1),
                bias=False,
                conv_cfg=dict(type='Conv2d'),
                norm_cfg=dict(type='BN', requires_grad=True))
        self.dropout_ratio = dropout_ratio
        self.init_std = init_std
        if self.dropout_ratio != 0:
            self.dropout = nn.Dropout(p=self.dropout_ratio)
        else:
            self.dropout = None
        self.f1_fc = nn.Linear(self.in_channels * 2, self.num_classes)
        self.f2_fc = nn.Linear(self.in_channels * 2, self.num_classes)
        self.f3_fc = nn.Linear(self.in_channels * 2, self.num_classes)
        self.avg_pool = nn.AdaptiveAvgPool3d((1, 1, 1))

    def init_weights(self):
        """Initiate the parameters from scratch."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                normal_init(m, std=self.init_std)
            if isinstance(m, nn.Conv3d):
                xavier_init(m, distribution='uniform')
            if isinstance(m, nn.BatchNorm3d):
                constant_init(m, 1)

    def exp_evidence(self, y):
        return torch.exp(torch.clamp(y, -10, 10))

    def edl_loss(self, func, alpha, y):
        S = torch.sum(alpha, dim=1, keepdim=True)
        loss = torch.sum(y * (func(S) - func(alpha)), dim=1, keepdim=True)
        return loss

    def kl_divergence(self, alpha, beta):
        # compute the negative KL divergence between two Dirichlet distribution
        S_alpha = torch.sum(alpha, dim=1, keepdim=True)
        S_beta = torch.sum(beta, dim=1, keepdim=True)
        lnA = torch.lgamma(S_alpha) - torch.sum(torch.lgamma(alpha), dim=1, keepdim=True)
        lnB = torch.lgamma(S_beta) - torch.sum(torch.lgamma(beta), dim=1, keepdim=True)
        # compute the digamma term
        dg_term = torch.digamma(alpha) - torch.digamma(S_alpha)
        # final KL divergence
        kl = lnA - lnB + torch.sum((alpha - beta) * dg_term, dim=1, keepdim=True)
        return kl

    def _kernel(self, X, sigma):
        X = X.view(len(X), -1)
        XX = X @ X.t()
        X_sqnorms = torch.diag(XX)
        X_L2 = -2 * XX + X_sqnorms.unsqueeze(1) + X_sqnorms.unsqueeze(0)
        gamma = 1 / (2 * sigma ** 2)

        kernel_XX = torch.exp(-gamma * X_L2)
        return kernel_XX

    def hsic_loss(self, input1, input2, unbiased=False):
        N = len(input1)
        if N < 4:
            return torch.tensor(0.0).to(input1.device)
        # we simply use the squared dimension of feature as the sigma for RBF kernel
        sigma_x = np.sqrt(input1.size()[1])
        sigma_y = np.sqrt(input2.size()[1])

        # compute the kernels
        kernel_XX = self._kernel(input1, sigma_x)
        kernel_YY = self._kernel(input2, sigma_y)

        if unbiased:
            """Unbiased estimator of Hilbert-Schmidt Independence Criterion
            Song, Le, et al. "Feature selection via dependence maximization." 2012.
            """
            tK = kernel_XX - torch.diag(torch.diag(kernel_XX))
            tL = kernel_YY - torch.diag(torch.diag(kernel_YY))
            hsic = (
                torch.trace(tK @ tL)
                + (torch.sum(tK) * torch.sum(tL) / (N - 1) / (N - 2))
                - (2 * torch.sum(tK, 0).dot(torch.sum(tL, 0)) / (N - 2))
            )
            loss = hsic if self.alternative else hsic / (N * (N - 3))
        else:
            """Biased estimator of Hilbert-Schmidt Independence Criterion
            Gretton, Arthur, et al. "Measuring statistical dependence with Hilbert-Schmidt norms." 2005.
            """
            KH = kernel_XX - kernel_XX.mean(0, keepdim=True)
            LH = kernel_YY - kernel_YY.mean(0, keepdim=True)
            loss = torch.trace(KH @ LH / (N - 1) ** 2)
        return loss

    def forward(self, x, num_segs=None, target=None, **kwargs):
        """Defines the computation performed at every call.

        Args:
            x (torch.Tensor): The input data. (B, 1024, 8, 14, 14)

        Returns:
            torch.Tensor: The classification scores for input samples.
        """
        feat = x.clone() if isinstance(x, torch.Tensor) else x[-2].clone()
        if len(feat.size()) == 4:  # for 2D recognizer
            assert num_segs is not None
            feat = feat.view((-1, num_segs) + feat.size()[1:]).transpose(1, 2).contiguous()
        # one-hot embedding for the target
        y = torch.eye(self.num_classes).to(feat.device)
        y = y[target]
        losses = dict()

        # f1_Conv3D(x)
        x = self.f1_conv3d(feat)  # (B, 2048, 8, 7, 7)
        feat_unbias = self.avg_pool(x).squeeze(-1).squeeze(-1).squeeze(-1)
        x = self.dropout(feat_unbias)
        x = self.f1_fc(x)
        alpha_unbias = self.exp_evidence(x) + 1
        # minimize the edl losses
        loss_cls1 = self.edl_loss(torch.log, alpha_unbias, y)
        losses.update({'loss_unbias_cls': loss_cls1})

        loss_hsic_f, loss_hsic_g = torch.zeros_like(loss_cls1), torch.zeros_like(loss_cls1)
        if self.bias_input:
            # f2_Conv3D(x)
            feat_shuffle = feat[:, :, torch.randperm(feat.size()[2])]
            x = self.f2_conv3d(feat_shuffle)  # (B, 2048, 8, 7, 7)
            feat_bias1 = self.avg_pool(x).squeeze(-1).squeeze(-1).squeeze(-1)
            x = self.dropout(feat_bias1)
            x = self.f2_fc(x)
            alpha_bias1 = self.exp_evidence(x) + 1
            # minimize the edl losses
            loss_cls2 = self.edl_loss(torch.log, alpha_bias1, y)
            losses.update({'loss_bias1_cls': loss_cls2})
            if self.alternative:
                # minimize HSIC w.r.t. feat_unbias, and maximize HSIC w.r.t. feat_bias1
                loss_hsic_f += self.hsic_factor * self.hsic_loss(feat_unbias, feat_bias1.detach(), unbiased=True) 
                loss_hsic_g += - self.hsic_factor * self.hsic_loss(feat_unbias.detach(), feat_bias1, unbiased=True)
            else:
                # maximize HSIC 
                loss_hsic1 = -1.0 * self.hsic_loss(alpha_unbias, alpha_bias1)
                losses.update({"loss_bias1_hsic": loss_hsic1})

        if self.bias_network:
            # f3_Conv2D(x)
            B, C, T, H, W = feat.size()
            feat_reshape = feat.permute(0, 2, 1, 3, 4).contiguous().view(-1, C, H, W)  # (B*T, C, H, W)
            x = self.f3_conv2d(feat_reshape)  # (64, 2048, 7, 7)
            x = x.view(B, T, x.size(-3), x.size(-2), x.size(-1)).permute(0, 2, 1, 3, 4)  # (B, 2048, 8, 7, 7)
            feat_bias2 = self.avg_pool(x).squeeze(-1).squeeze(-1).squeeze(-1)
            x = self.dropout(feat_bias2)
            x = self.f3_fc(x)
            alpha_bias2 = self.exp_evidence(x) + 1
            # minimize the edl losses
            loss_cls3 = self.edl_loss(torch.log, alpha_bias2, y)
            losses.update({'loss_bias2_cls': loss_cls3})
            if self.alternative:
                # minimize HSIC w.r.t. feat_unbias, and maximize HSIC w.r.t. feat_bias2
                loss_hsic_f += self.hsic_factor * self.hsic_loss(feat_unbias, feat_bias2.detach(), unbiased=True)
                loss_hsic_g += - self.hsic_factor * self.hsic_loss(feat_unbias.detach(), feat_bias2, unbiased=True)
            else:
                # maximize HSIC 
                loss_hsic2 = -1.0 * self.hsic_loss(alpha_unbias, alpha_bias2)
                losses.update({"loss_bias2_hsic": loss_hsic2})
        
        if self.alternative:
            # Here, we use odd iterations for minimizing hsic_f, and use even iterations for maximizing hsic_g
            assert 'iter' in kwargs, "iter number is missing!"
            loss_mask = kwargs['iter'] % 2
            loss_hsic = loss_mask * loss_hsic_f + (1 - loss_mask) * loss_hsic_g
            losses.update({'loss_hsic': loss_hsic})
            
        for k, v in losses.items():
            losses.update({k: v * self.loss_factor})
        return losses
