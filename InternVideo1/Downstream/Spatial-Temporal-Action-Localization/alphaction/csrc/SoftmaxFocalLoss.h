#pragma once

#include "cpu/vision.h"

#ifdef WITH_CUDA
#include "cuda/vision.h"
#endif

// Interface for Python
std::tuple<at::Tensor, at::Tensor> SoftmaxFocalLoss_forward(
		const at::Tensor& logits,
                const at::Tensor& targets,
		const float gamma,
		const float alpha) {
  if (logits.is_cuda()) {
#ifdef WITH_CUDA
    return SoftmaxFocalLoss_forward_cuda(logits, targets, gamma, alpha);
#else
    AT_ERROR("Not compiled with GPU support");
#endif
  }
  AT_ERROR("Not implemented on the CPU");
}

at::Tensor SoftmaxFocalLoss_backward(
			     const at::Tensor& logits,
                             const at::Tensor& targets,
                 const at::Tensor& P,
			     const at::Tensor& d_losses,
			     const float gamma,
			     const float alpha) {
  if (logits.is_cuda()) {
#ifdef WITH_CUDA
    return SoftmaxFocalLoss_backward_cuda(logits, targets, P, d_losses, gamma, alpha);
#else
    AT_ERROR("Not compiled with GPU support");
#endif
  }
  AT_ERROR("Not implemented on the CPU");
}