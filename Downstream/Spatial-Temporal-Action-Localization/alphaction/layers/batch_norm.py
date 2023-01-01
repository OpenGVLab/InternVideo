import torch
from torch import nn


class _FrozenBatchNorm(nn.Module):
    def __init__(self, num_features, eps=1e-5, affine=True, track_running_stats=True):
        super(_FrozenBatchNorm, self).__init__()
        self.num_features = num_features
        self.eps = eps
        self.affine = affine
        self.track_running_stats = track_running_stats
        if self.affine:
            self.register_buffer("weight", torch.Tensor(num_features))
            self.register_buffer("bias", torch.Tensor(num_features))
        else:
            self.register_buffer("weight", None)
            self.register_buffer("bias", None)
        if self.track_running_stats:
            self.register_buffer('running_mean', torch.zeros(num_features))
            self.register_buffer('running_var', torch.ones(num_features))
        else:
            self.register_parameter('running_mean', None)
            self.register_parameter('running_var', None)
        self.reset_parameters()

    def reset_running_stats(self):
        if self.track_running_stats:
            self.running_mean.zero_()
            self.running_var.fill_(1)

    def reset_parameters(self):
        self.reset_running_stats()
        if self.affine:
            self.weight.data.uniform_()
            self.bias.data.zero_()

    def _check_input_dim(self, input):
        raise NotImplementedError

    def forward(self, input):
        self._check_input_dim(input)
        view_shape = (1, self.num_features) + (1,) * (input.dim() - 2)

        if self.track_running_stats:
            scale = self.weight / (self.running_var + self.eps).sqrt()
            bias = self.bias - self.running_mean * scale
        else:
            scale = self.weight
            bias = self.bias

        return scale.view(*view_shape) * input + bias.view(*view_shape)

    def extra_repr(self):
        return '{num_features}, eps={eps}, affine={affine}, ' \
               'track_running_stats={track_running_stats}'.format(**self.__dict__)

    def _load_from_state_dict(self, state_dict, prefix, metadata, strict,
                              missing_keys, unexpected_keys, error_msgs):
        num_batches_tracked_key = prefix + 'num_batches_tracked'
        if num_batches_tracked_key in state_dict:
            del state_dict[num_batches_tracked_key]
        super(_FrozenBatchNorm, self)._load_from_state_dict(
            state_dict, prefix, metadata, strict,
            missing_keys, unexpected_keys, error_msgs)


class FrozenBatchNorm1d(_FrozenBatchNorm):
    def _check_input_dim(self, input):
        if input.dim() != 2 and input.dim() != 3:
            raise ValueError('expected 2D or 3D input (got {}D input)'
                             .format(input.dim()))


class FrozenBatchNorm2d(_FrozenBatchNorm):
    def _check_input_dim(self, input):
        if input.dim() != 4:
            raise ValueError('expected 4D input (got {}D input)'
                             .format(input.dim()))


class FrozenBatchNorm3d(_FrozenBatchNorm):
    def _check_input_dim(self, input):
        if input.dim() != 5:
            raise ValueError('expected 5D input (got {}D input)'
                             .format(input.dim()))
