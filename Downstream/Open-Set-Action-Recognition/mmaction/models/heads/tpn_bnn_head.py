import torch.nn as nn

from ..registry import HEADS
from .tsn_head import TSNHead
from ..builder import build_loss
from .bnn import BayesianPredictor, get_uncertainty


@HEADS.register_module()
class TPNBNNHead(TSNHead):
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

    def __init__(self, compute_uncertainty=False, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.compute_uncertainty = compute_uncertainty

        # use bnn classification head
        self.bnn_cls = BayesianPredictor(self.in_channels, self.num_classes)
        self.bnn_loss = self.loss_cls

        if self.spatial_type == 'avg':
            # use `nn.AdaptiveAvgPool3d` to adaptively match the in_channels.
            self.avg_pool3d = nn.AdaptiveAvgPool3d((1, 1, 1))
        else:
            self.avg_pool3d = None

        self.avg_pool2d = None
        

    def forward(self, x, num_segs=None, npass=2, testing=False):
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
        outputs, log_priors, log_variational_posteriors = self.bnn_cls(x, npass=npass, testing=testing)

        # gather output dictionary
        output = outputs.mean(0)
        log_prior = log_priors.mean()
        log_variational_posterior = log_variational_posteriors.mean()
        output_dict = {'pred_mean': output,
                       'log_prior': log_prior,
                       'log_posterior': log_variational_posterior}
        if self.compute_uncertainty:
            uncertain_alea, uncertain_epis = get_uncertainty(outputs)
            output_dict.update({'aleatoric': uncertain_alea,
                                'epistemic': uncertain_epis})
        return output_dict
