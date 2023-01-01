#pragma once
#include <torch/extension.h>

at::Tensor ROIAlign3d_forward_cuda(const at::Tensor& input,
                                 const at::Tensor& rois,
                                 const float spatial_scale,
                                 const int pooled_height,
                                 const int pooled_width,
                                 const int sampling_ratio);

at::Tensor ROIAlign3d_backward_cuda(const at::Tensor& grad,
                                  const at::Tensor& rois,
                                  const float spatial_scale,
                                  const int pooled_height,
                                  const int pooled_width,
                                  const int batch_size,
                                  const int channels,
                                  const int length,
                                  const int height,
                                  const int width,
                                  const int sampling_ratio);

std::tuple<at::Tensor, at::Tensor> ROIPool3d_forward_cuda(const at::Tensor& input,
                                const at::Tensor& rois,
                                const float spatial_scale,
                                const int pooled_height,
                                const int pooled_width);

at::Tensor ROIPool3d_backward_cuda(const at::Tensor& grad,
                                 const at::Tensor& input,
                                 const at::Tensor& rois,
                                 const at::Tensor& argmax,
                                 const float spatial_scale,
                                 const int pooled_height,
                                 const int pooled_width,
                                 const int batch_size,
                                 const int channels,
                                 const int length,
                                 const int height,
                                 const int width);


at::Tensor SigmoidFocalLoss_forward_cuda(
		const at::Tensor& logits,
        const at::Tensor& targets,
		const float gamma,
		const float alpha);

at::Tensor SigmoidFocalLoss_backward_cuda(
		const at::Tensor& logits,
        const at::Tensor& targets,
		const at::Tensor& d_losses,
		const float gamma,
		const float alpha);

std::tuple<at::Tensor, at::Tensor> SoftmaxFocalLoss_forward_cuda(
		const at::Tensor& logits,
        const at::Tensor& targets,
		const float gamma,
		const float alpha);

at::Tensor SoftmaxFocalLoss_backward_cuda(
		const at::Tensor& logits,
        const at::Tensor& targets,
        const at::Tensor& P,
		const at::Tensor& d_losses,
		const float gamma,
		const float alpha);