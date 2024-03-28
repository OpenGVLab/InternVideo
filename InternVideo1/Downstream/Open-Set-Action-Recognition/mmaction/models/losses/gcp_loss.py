import torch
import torch.nn.functional as F

from ..registry import LOSSES
from .base import BaseWeightedLoss
from ...core import top_k_accuracy


@LOSSES.register_module()
class GCPLoss(BaseWeightedLoss):
    """Reciprocal Point Learning Loss."""
    def __init__(self, temperature=1, weight_pl=0.1, radius_init=1):
        super().__init__()
        self.temperature = temperature
        self.weight_pl = weight_pl

    def _forward(self, head_outs, labels, **kwargs):
        """Forward function.

        Args:
            head_outs (torch.Tensor): outputs of the RPL head
            label (torch.Tensor): The ground truth label.
            kwargs: Any keyword argument to be used to calculate
                CrossEntropy loss.

        Returns:
            torch.Tensor: The returned CrossEntropy loss.
        """
        dist, feature, centers = head_outs['dist'], head_outs['feature'], head_outs['centers']
        if labels.shape == torch.Size([]):
            labels = labels.unsqueeze(0)
        # compute losses
        logits = F.softmax(dist, dim=1)
        loss_closed = F.cross_entropy(dist / self.temperature, labels, **kwargs)
        center_batch = centers[labels, :]
        loss_r = F.mse_loss(feature, center_batch) / 2
        # gather losses
        losses = {'loss_cls': loss_closed, 'loss_open': self.weight_pl * loss_r}

        # compute top-K accuracy using CPU numpy
        top_k_acc = top_k_accuracy(logits.detach().cpu().numpy(),
                                       labels.detach().cpu().numpy(), (1, 5))
        losses.update({'top1_acc': torch.tensor(top_k_acc[0], device=dist.device)})
        losses.update({'top5_acc': torch.tensor(top_k_acc[1], device=dist.device)})

        return losses