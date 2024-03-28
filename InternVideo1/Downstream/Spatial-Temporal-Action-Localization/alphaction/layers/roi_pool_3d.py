from torch import nn
from torch.autograd import Function
from torch.autograd.function import once_differentiable
from torch.nn.modules.utils import _pair

# import alphaction._custom_cuda_ext as _C


class _ROIPool3d(Function):
    @staticmethod
    def forward(ctx, input, roi, output_size, spatial_scale):
        ctx.output_size = _pair(output_size)
        ctx.spatial_scale = spatial_scale
        ctx.input_shape = input.size()
        output, argmax = _C.roi_pool_3d_forward(
            input, roi, spatial_scale, output_size[0], output_size[1]
        )
        ctx.save_for_backward(input, roi, argmax)
        return output

    @staticmethod
    @once_differentiable
    def backward(ctx, grad_output):
        input, rois, argmax = ctx.saved_tensors
        output_size = ctx.output_size
        spatial_scale = ctx.spatial_scale
        bs, ch, l, h, w = ctx.input_shape
        grad_input = _C.roi_pool_3d_backward(
            grad_output,
            input,
            rois,
            argmax,
            spatial_scale,
            output_size[0],
            output_size[1],
            bs,
            ch,
            l,
            h,
            w,
        )
        return grad_input, None, None, None


roi_pool_3d = _ROIPool3d.apply


class ROIPool3d(nn.Module):
    def __init__(self, output_size, spatial_scale):
        super(ROIPool3d, self).__init__()
        self.output_size = output_size
        self.spatial_scale = spatial_scale

    def forward(self, input, rois):
        return roi_pool_3d(input, rois, self.output_size, self.spatial_scale)

    def __repr__(self):
        tmpstr = self.__class__.__name__ + "("
        tmpstr += "output_size=" + str(self.output_size)
        tmpstr += ", spatial_scale=" + str(self.spatial_scale)
        tmpstr += ")"
        return tmpstr