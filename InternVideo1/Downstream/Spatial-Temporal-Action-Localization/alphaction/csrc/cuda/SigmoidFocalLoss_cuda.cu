// This file is modified from https://github.com/facebookresearch/maskrcnn-benchmark/blob/master/maskrcnn_benchmark/csrc/cuda/SigmoidFocalLoss_cuda.cu
// Jiajun Tang
// yelantingfeng@sjtu.edu.cn
#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>

#include <THC/THC.h>
#include <THC/THCAtomics.cuh>
#include <THC/THCDeviceUtils.cuh>

#include <cfloat>

// TODO make it in a common file
#define CUDA_1D_KERNEL_LOOP(i, n)                            \
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; \
       i += blockDim.x * gridDim.x)


template <typename T>
__global__ void SigmoidFocalLossForward(const int nthreads,
    const T* logits,
    const T* targets,
    const int num_classes,
    const float gamma,
    const float alpha,
    T* losses) {
  CUDA_1D_KERNEL_LOOP(i, nthreads) {

    // Decide it is positive or negative case.
    T c1 = targets[i];
    T c2 = (1.0 - c1);

    // alpha flag.
    T af1 = (alpha >= 0);
    T af2 = (1.0 - af1);

    T zn = (1.0 - alpha) * af1 + af2;
    T zp = (alpha) * af1 + af2;

    // p = 1. / 1. + expf(-x); p = sigmoid(x)
    T  p = 1. / (1. + expf(-logits[i]));

    // (1-p)**gamma * log(p) where
    T term1 = powf((1. - p), gamma) * logf(max(p, FLT_MIN));

    // p**gamma * log(1-p)
    T term2 = powf(p, gamma) *
            (-1. * logits[i] * (logits[i] >= 0) -
             logf(1. + expf(logits[i] - 2. * logits[i] * (logits[i] >= 0))));

    losses[i] = 0.0;
    losses[i] += -c1 * term1 * zp;
    losses[i] += -c2 * term2 * zn;

  } // CUDA_1D_KERNEL_LOOP
} // SigmoidFocalLossForward


template <typename T>
__global__ void SigmoidFocalLossBackward(const int nthreads,
                const T* logits,
                const T* targets,
                const T* d_losses,
                const int num_classes,
                const float gamma,
                const float alpha,
                T* d_logits) {
  CUDA_1D_KERNEL_LOOP(i, nthreads) {

    // Decide it is positive or negative case.
    T c1 = targets[i];
    T c2 = (1 - c1);

    // alpha flag.
    T af1 = (alpha >= 0);
    T af2 = (1.0 - af1);

    T zn = (1.0 - alpha) * af1 + af2;
    T zp = (alpha) * af1 + af2;
    // p = 1. / 1. + expf(-x); p = sigmoid(x)
    T  p = 1. / (1. + expf(-logits[i]));

    // (1-p)**g * (1 - p - g*p*log(p)
    T term1 = powf((1. - p), gamma) *
                      (1. - p - (p * gamma * logf(max(p, FLT_MIN))));

    // (p**g) * (g*(1-p)*log(1-p) - p)
    T term2 = powf(p, gamma) *
                  ((-1. * logits[i] * (logits[i] >= 0) -
                      logf(1. + expf(logits[i] - 2. * logits[i] * (logits[i] >= 0)))) *
                      (1. - p) * gamma - p);
    d_logits[i] = 0.0;
    d_logits[i] += -c1 * term1 * zp;
    d_logits[i] += -c2 * term2 * zn;
    d_logits[i] = d_logits[i] * d_losses[i];

  } // CUDA_1D_KERNEL_LOOP
} // SigmoidFocalLossBackward


at::Tensor SigmoidFocalLoss_forward_cuda(
		const at::Tensor& logits,
                const at::Tensor& targets,
		const float gamma,
		const float alpha) {
  AT_ASSERTM(logits.is_cuda(), "logits must be a CUDA tensor");
  AT_ASSERTM(targets.is_cuda(), "targets must be a CUDA tensor");
  AT_ASSERTM(logits.dim() == 2, "logits should be NxClass");
  AT_ASSERTM(targets.dim() == 2, "targets should be NxClass");
  AT_ASSERTM(logits.size(0) == targets.size(0) && logits.size(1) == targets.size(1),
      "targets should have exactly the same shape with logits.");

  const int num_samples = logits.size(0);
  const int num_classes = logits.size(1);

  auto losses = at::empty({num_samples, num_classes}, logits.options());
  auto losses_size = static_cast<long>(num_samples) * num_classes;
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  dim3 grid(std::min(THCCeilDiv(losses_size, 512L), 4096L));
  dim3 block(512);

  if (losses.numel() == 0) {
    THCudaCheck(cudaGetLastError());
    return losses;
  }

  AT_DISPATCH_FLOATING_TYPES(logits.scalar_type(), "SigmoidFocalLoss_forward", [&] {
    SigmoidFocalLossForward<scalar_t><<<grid, block, 0, stream>>>(
         losses_size,
         logits.contiguous().data_ptr<scalar_t>(),
	 targets.contiguous().data_ptr<scalar_t>(),
         num_classes,
	 gamma,
	 alpha,
         losses.data_ptr<scalar_t>());
  });
  THCudaCheck(cudaGetLastError());
  return losses;
}


at::Tensor SigmoidFocalLoss_backward_cuda(
		const at::Tensor& logits,
                const at::Tensor& targets,
		const at::Tensor& d_losses,
		const float gamma,
		const float alpha) {
  AT_ASSERTM(logits.is_cuda(), "logits must be a CUDA tensor");
  AT_ASSERTM(targets.is_cuda(), "targets must be a CUDA tensor");
  AT_ASSERTM(d_losses.is_cuda(), "d_losses must be a CUDA tensor");

  AT_ASSERTM(logits.dim() == 2, "logits should be NxClass");
  AT_ASSERTM(targets.dim() == 2, "targets should be NxClass");
  AT_ASSERTM(logits.size(0) == targets.size(0) && logits.size(1) == targets.size(1),
      "targets should have exactly the same shape with logits.");

  const int num_samples = logits.size(0);
  const int num_classes = logits.size(1);

  auto d_logits = at::zeros({num_samples, num_classes}, logits.options());
  auto d_logits_size = static_cast<long>(num_samples) * num_classes;
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  dim3 grid(std::min(THCCeilDiv(d_logits_size, 512L), 4096L));
  dim3 block(512);

  if (d_logits.numel() == 0) {
    THCudaCheck(cudaGetLastError());
    return d_logits;
  }

  AT_DISPATCH_FLOATING_TYPES(logits.scalar_type(), "SigmoidFocalLoss_backward", [&] {
    SigmoidFocalLossBackward<scalar_t><<<grid, block, 0, stream>>>(
         d_logits_size,
         logits.contiguous().data_ptr<scalar_t>(),
	 targets.contiguous().data_ptr<scalar_t>(),
	 d_losses.contiguous().data_ptr<scalar_t>(),
         num_classes,
	 gamma,
	 alpha,
         d_logits.data_ptr<scalar_t>());
  });

  THCudaCheck(cudaGetLastError());
  return d_logits;
}

