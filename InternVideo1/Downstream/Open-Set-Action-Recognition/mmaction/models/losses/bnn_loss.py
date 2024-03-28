import torch
import torch.nn as nn
import torch.nn.functional as F

from ..registry import LOSSES
from ...core import top_k_accuracy

@LOSSES.register_module()
class BayesianNNLoss(nn.Module):
    """Bayesian NN Loss."""

    def forward(self, cls_score, labels, output_dict, beta=1.0, **kwargs):
        """Forward function.

        Args:
            cls_score (torch.Tensor): The class score.
            label (torch.Tensor): The ground truth label.
            prior (torch.Tensor): The log prior
            posterior (torch.Tensor): The log variational posterior
            kwargs: Any keyword argument to be used to calculate
                Bayesian NN loss.

        Returns:
            torch.Tensor: The returned Bayesian NN loss.
        """
        losses = dict()
        if labels.shape == torch.Size([]):
            labels = labels.unsqueeze(0)

        # negative log-likelihood (BCE loss)
        loss_cls = F.cross_entropy(cls_score, labels, **kwargs)
        # parse the output
        log_prior = output_dict['log_prior']
        log_posterior = output_dict['log_posterior']

        # complexity regularizer
        loss_complexity = beta * (log_posterior - log_prior)
        # total loss
        loss = loss_cls + loss_complexity
        # accuracy metrics
        top_k_acc = top_k_accuracy(cls_score.detach().cpu().numpy(),
                                    labels.detach().cpu().numpy(), (1, 5))
        losses = {'loss_cls': loss_cls, 'loss_complexity': loss_complexity,  # items to be backwarded
                  'LOSS_total': loss,  # items for monitoring
                  'log_posterior': beta * log_posterior,
                  'log_prior': beta * log_prior, 
                  'top1_acc': torch.tensor(top_k_acc[0], device=cls_score.device),
                  'top5_acc': torch.tensor(top_k_acc[1], device=cls_score.device)
                  }
        if 'aleatoric' in output_dict: losses.update({'aleatoric': output_dict['aleatoric']})
        if 'epistemic' in output_dict: losses.update({'epistemic': output_dict['epistemic']})
        return losses


