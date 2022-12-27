import torch
import torch.nn.functional as F
import numpy as np

from ..registry import LOSSES
from .base import BaseWeightedLoss


@LOSSES.register_module()
class RebiasLoss(BaseWeightedLoss):
    """Rebias Loss."""
    def __init__(self, lambda_g=1.0, criteria='hsic'):
        super().__init__()
        self.lambda_g = lambda_g
        self.criteria = criteria

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
            tK = kernel_XX - torch.diag(kernel_XX)
            tL = kernel_YY - torch.diag(kernel_YY)
            hsic = (
                torch.trace(tK @ tL)
                + (torch.sum(tK) * torch.sum(tL) / (N - 1) / (N - 2))
                - (2 * torch.sum(tK, 0).dot(torch.sum(tL, 0)) / (N - 2))
            )
            loss = hsic / (N * (N - 3))
        else:
            """Biased estimator of Hilbert-Schmidt Independence Criterion
            Gretton, Arthur, et al. "Measuring statistical dependence with Hilbert-Schmidt norms." 2005.
            """
            KH = kernel_XX - kernel_XX.mean(0, keepdim=True)
            LH = kernel_YY - kernel_YY.mean(0, keepdim=True)
            loss = torch.trace(KH @ LH / (N - 1) ** 2)
        return loss

    def cosine_loss(self, input1, input2):
        # normalize the inputs with L2-norm
        norm1 = F.normalize(input1, dim=1, p=2)
        norm2 = F.normalize(input2, dim=1, p=2)
        # cosine distance
        cos_batch = torch.bmm(norm1.unsqueeze(1), norm2.unsqueeze(2)).squeeze(-1).squeeze(-1)
        loss = torch.mean(torch.abs(cos_batch))
        return loss


    def _forward(self, x, xs, y, ys, label, **kwargs):
        """Forward function.
        Returns:
            torch.Tensor: The returned Rebias loss.
        """
        # L(f)
        loss_f = F.cross_entropy(y, label, **kwargs)
        # L(g)
        loss_g = self.lambda_g * F.cross_entropy(ys, label, **kwargs)
        losses = {'loss_f': loss_f, 'loss_g': loss_g}
        # all losses
        if self.criteria == 'hsic':
            # negative HSIC loss
            loss_hsic = - self.hsic_loss(x, xs)  # small returned value means high dependency
            losses.update({'loss_hsic': loss_hsic})
        elif self.criteria == 'cosine':
            loss_cos = self.cosine_loss(x, xs)  # large returned value means high dependency
            losses.update({'loss_cos': loss_cos})
        else:
            raise NotImplementedError
        
        return losses