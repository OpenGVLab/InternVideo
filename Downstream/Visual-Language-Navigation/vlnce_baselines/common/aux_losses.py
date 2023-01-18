import torch


class _AuxLosses:
    def __init__(self):
        self._losses = {}
        self._loss_alphas = {}
        self._is_active = False

    def clear(self):
        self._losses.clear()
        self._loss_alphas.clear()

    def register_loss(self, name, loss, alpha=1.0):
        assert self.is_active()
        assert name not in self._losses

        self._losses[name] = loss
        self._loss_alphas[name] = alpha

    def get_loss(self, name):
        return self._losses[name]

    def reduce(self, mask):
        assert self.is_active()
        total = torch.tensor(0.0).cuda()

        for k in self._losses.keys():
            k_loss = torch.masked_select(self._losses[k], mask).mean()
            total = total + self._loss_alphas[k] * k_loss

        return total

    def is_active(self):
        return self._is_active

    def activate(self):
        self._is_active = True

    def deactivate(self):
        self._is_active = False


AuxLosses = _AuxLosses()
