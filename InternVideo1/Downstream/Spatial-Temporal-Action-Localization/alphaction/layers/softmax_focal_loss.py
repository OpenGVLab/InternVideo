import torch
from torch import nn
from torch.autograd import Function
from torch.autograd.function import once_differentiable

# import alphaction._custom_cuda_ext as _C


class _SoftmaxFocalLoss(Function):
    @staticmethod
    def forward(ctx, logits, targets, gamma, alpha):
        ctx.gamma = gamma
        ctx.alpha = alpha

        losses, P = _C.softmax_focalloss_forward(
            logits, targets, gamma, alpha
        )
        ctx.save_for_backward(logits, targets, P)
        return losses

    @staticmethod
    @once_differentiable
    def backward(ctx, d_loss):
        logits, targets, P = ctx.saved_tensors
        gamma = ctx.gamma
        alpha = ctx.alpha
        d_logits = _C.softmax_focalloss_backward(
            logits, targets, P, d_loss, gamma, alpha
        )
        return d_logits, None, None, None


def softmax_focal_loss(logits, targets, gamma, alpha, reduction='mean'):
    assert reduction in ["none", "mean", "sum"], "Unsupported reduction type \"{}\"".format(reduction)
    logits = logits.float()
    targets = targets.int()

    ret = _SoftmaxFocalLoss.apply(logits, targets, gamma, alpha)
    if reduction != "none":
        ret = torch.mean(ret) if reduction == "mean" else torch.sum(ret)

    return ret


class SoftmaxFocalLoss(nn.Module):
    def __init__(self, gamma, alpha, reduction="mean"):
        super(SoftmaxFocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction

    def forward(self, logits, targets):
        loss = softmax_focal_loss(logits, targets, self.gamma, self.alpha, self.reduction)
        return loss

    def __repr__(self):
        tmpstr = self.__class__.__name__ + "("
        tmpstr += "gamma=" + str(self.gamma)
        tmpstr += ", alpha=" + str(self.alpha)
        tmpstr += ")"
        return tmpstr
