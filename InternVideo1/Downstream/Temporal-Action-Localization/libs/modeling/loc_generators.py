import torch
from torch import nn
from torch.nn import functional as F

from .models import register_generator


class BufferList(nn.Module):
    """
    Similar to nn.ParameterList, but for buffers

    Taken from https://github.com/facebookresearch/detectron2/blob/master/detectron2/modeling/anchor_generator.py
    """

    def __init__(self, buffers):
        super().__init__()
        for i, buffer in enumerate(buffers):
            # Use non-persistent buffer so the values are not saved in checkpoint
            self.register_buffer(str(i), buffer, persistent=False)

    def __len__(self):
        return len(self._buffers)

    def __iter__(self):
        return iter(self._buffers.values())

@register_generator('point')
class PointGenerator(nn.Module):
    """
        A generator for temporal "points"

        max_seq_len can be much larger than the actual seq length
    """
    def __init__(
        self,
        max_seq_len,        # max sequence length that the generator will buffer
        fpn_levels,         # number of fpn levels
        scale_factor,       # scale factor between two fpn levels
        regression_range,   # regression range (on feature grids)
        use_offset=False    # if to align the points at grid centers
    ):
        super().__init__()
        # sanity check, # fpn levels and length divisible
        assert len(regression_range) == fpn_levels
        assert max_seq_len % scale_factor**(fpn_levels - 1) == 0

        # save params
        self.max_seq_len = max_seq_len
        self.fpn_levels = fpn_levels
        self.scale_factor = scale_factor
        self.regression_range = regression_range
        self.use_offset = use_offset

        # generate all points and buffer the list
        self.buffer_points = self._generate_points()

    def _generate_points(self):
        points_list = []
        # loop over all points at each pyramid level
        for l in range(self.fpn_levels):
            stride = self.scale_factor ** l
            reg_range = torch.as_tensor(
                self.regression_range[l], dtype=torch.float)
            fpn_stride = torch.as_tensor(stride, dtype=torch.float)
            points = torch.arange(0, self.max_seq_len, stride)[:, None]
            # add offset if necessary (not in our current model)
            if self.use_offset:
                points += 0.5 * stride
            # pad the time stamp with additional regression range / stride
            reg_range = reg_range[None].repeat(points.shape[0], 1)
            fpn_stride = fpn_stride[None].repeat(points.shape[0], 1)
            # size: T x 4 (ts, reg_range, stride)
            points_list.append(torch.cat((points, reg_range, fpn_stride), dim=1))

        return BufferList(points_list)

    def forward(self, feats):
        # feats will be a list of torch tensors
        assert len(feats) == self.fpn_levels
        pts_list = []
        feat_lens = [feat.shape[-1] for feat in feats]
        for feat_len, buffer_pts in zip(feat_lens, self.buffer_points):
            assert feat_len <= buffer_pts.shape[0], "Reached max buffer length for point generator"
            pts = buffer_pts[:feat_len, :]
            pts_list.append(pts)
        return pts_list
