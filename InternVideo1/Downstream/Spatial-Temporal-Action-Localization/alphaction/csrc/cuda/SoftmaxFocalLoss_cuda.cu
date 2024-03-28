// This file is modified from https://github.com/pytorch/pytorch/blob/master/modules/detectron/softmax_focal_loss_op.cu
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
__global__ void SpatialSoftmaxForward(const int nthreads,
    const T* Xdata,
    T* Pdata,
    const int num_classes) {
  CUDA_1D_KERNEL_LOOP(i, nthreads) {
    // Subtract max on each cell for numerical reasons
    T max_val = -FLT_MAX;
    for(int c = i * num_classes; c < (i + 1) * num_classes; ++c){
        max_val = max(max_val, Xdata[c]);
    }
    // Exponentiate
    T expsum = 0.0;
    for(int c = i * num_classes; c < (i + 1) * num_classes; ++c){
        T expx = exp(Xdata[c] - max_val);
        Pdata[c] = expx;
        expsum += expx;
    }
    // Normalize
    for(int c = i * num_classes; c < (i + 1) * num_classes; ++c){
        Pdata[c] /= expsum;
    }
  }
}

template <typename T>
__global__ void SoftmaxFocalLossForward(const int nthreads,
    const T* Pdata,
    const int* targets,
    const int num_classes,
    const float gamma,
    const float alpha,
    T* losses) {
  CUDA_1D_KERNEL_LOOP(n, nthreads) {

    const int label = static_cast<int>(targets[n]);

    // alpha flag.
    T af1 = (alpha >= 0);
    T af2 = (1.0 - af1);

    T z = ((label == 0) * (1 - alpha) +
          (label >= 1) * alpha) * af1 + af2;

    losses[n] = 0.0;
    if (label >= 0){
        int idx = n * num_classes + label;
        losses[n] =
          -(pow(1.0 - Pdata[idx], gamma) *
          log(max(Pdata[idx], FLT_MIN))) * z;
    }
  } // CUDA_1D_KERNEL_LOOP
} // SoftmaxFocalLossForward

template <typename T>
__global__ void SoftmaxFocalLossBackwardWeight(const int nthreads,
                const T* Pdata,
                const int* targets,
                const int num_classes,
                const float gamma,
                const float alpha,
                T* buff) {
  CUDA_1D_KERNEL_LOOP(n, nthreads) {

    const int label = static_cast<int>(targets[n]);

    // alpha flag.
    T af1 = (alpha >= 0);
    T af2 = (1.0 - af1);

    T z = ((label == 0) * (1 - alpha) +
          (label >= 1) * alpha) * af1 + af2;

    buff[n] = 0.0;
    if (label >= 0) {
      int idx = n * num_classes + label;
      T onemp = 1. - Pdata[idx];
      T p = Pdata[idx];
      buff[n] =
          (-pow(onemp, gamma) +
          gamma * pow(onemp, gamma - 1) * p * log(max(p, FLT_MIN))) * z;
    }
  }
}

template <typename T>
__global__ void SoftmaxFocalLossBackward(const int nthreads,
                const T* Pdata,
                const int* targets,
                const T* d_losses,
                const T* buff,
                const int num_classes,
                const float gamma,
                const float alpha,
                T* d_logits) {
  CUDA_1D_KERNEL_LOOP(i, nthreads) {

    int n = i / num_classes;
    int c = i % num_classes;
    T d_loss = d_losses[n];

    const int label = static_cast<int>(targets[n]);

    T c1 = (label >= 0) * 1.0;
    T c2 = (label == c) * 1.0;
    d_logits[i] = c1 * d_loss * buff[n] * (c2 - Pdata[i]);
  } // CUDA_1D_KERNEL_LOOP
} // SoftmaxFocalLossBackward


std::tuple<at::Tensor, at::Tensor> SoftmaxFocalLoss_forward_cuda(
		const at::Tensor& logits,
                const at::Tensor& targets,
		const float gamma,
		const float alpha) {
  AT_ASSERTM(logits.is_cuda(), "logits must be a CUDA tensor");
  AT_ASSERTM(targets.is_cuda(), "targets must be a CUDA tensor");
  AT_ASSERTM(logits.dim() == 2, "logits should be NxClass");
  AT_ASSERTM(targets.dim() == 1, "targets should be N");
  AT_ASSERTM(logits.size(0) == targets.size(0),
      "dim(0) of targets should be the same as dim(0) of logits.");

  const int num_samples = logits.size(0);
  const int num_classes = logits.size(1);

  auto losses = at::empty({num_samples}, logits.options());
  auto losses_size = static_cast<long>(num_samples);
  auto P = at::empty({num_samples, num_classes}, logits.options());
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  dim3 grid(std::min(THCCeilDiv(losses_size, 512L), 4096L));
  dim3 block(512);

  if (losses.numel() == 0) {
    THCudaCheck(cudaGetLastError());
    return std::make_tuple(losses, P);
  }

  AT_DISPATCH_FLOATING_TYPES(logits.scalar_type(), "SpatialSoftmax_forward", [&] {
    SpatialSoftmaxForward<scalar_t><<<grid, block, 0, stream>>>(
      losses_size,
      logits.contiguous().data_ptr<scalar_t>(),
      P.data_ptr<scalar_t>(),
      num_classes);
  });

  AT_DISPATCH_FLOATING_TYPES(logits.scalar_type(), "SoftmaxFocalLoss_forward", [&] {
    SoftmaxFocalLossForward<scalar_t><<<grid, block, 0, stream>>>(
      losses_size,
      P.data_ptr<scalar_t>(),
      targets.contiguous().data_ptr<int>(),
      num_classes,
      gamma,
      alpha,
      losses.data_ptr<scalar_t>());
  });
  THCudaCheck(cudaGetLastError());
  return std::make_tuple(losses, P);
}


at::Tensor SoftmaxFocalLoss_backward_cuda(
		const at::Tensor& logits,
                const at::Tensor& targets,
    const at::Tensor& P,
		const at::Tensor& d_losses,
		const float gamma,
		const float alpha) {
  AT_ASSERTM(logits.is_cuda(), "logits must be a CUDA tensor");
  AT_ASSERTM(targets.is_cuda(), "targets must be a CUDA tensor");
  AT_ASSERTM(d_losses.is_cuda(), "d_losses must be a CUDA tensor");

  AT_ASSERTM(logits.dim() == 2, "logits should be NxClass");
  AT_ASSERTM(targets.dim() == 1, "targets should be N");
  AT_ASSERTM(logits.size(0) == targets.size(0),
      "dim(0) of targets should be the same as dim(0) of logits.");

  const int num_samples = logits.size(0);
  const int num_classes = logits.size(1);

  auto buff = at::zeros({num_samples},logits.options());
  auto buff_size = static_cast<long>(num_samples);
  auto d_logits = at::zeros({num_samples, num_classes}, logits.options());
  auto d_logits_size = static_cast<long>(num_samples) * num_classes;
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  dim3 grid1(std::min(THCCeilDiv(buff_size, 512L), 4096L));
  dim3 grid2(std::min(THCCeilDiv(d_logits_size, 512L), 4096L));
  dim3 block(512);

  if (d_logits.numel() == 0) {
    THCudaCheck(cudaGetLastError());
    return d_logits;
  }

  AT_DISPATCH_FLOATING_TYPES(logits.scalar_type(), "SoftmaxFocalLoss_backwardWeight", [&] {
    SoftmaxFocalLossBackwardWeight<scalar_t><<<grid1, block, 0, stream>>>(
      buff_size,
      P.contiguous().data_ptr<scalar_t>(),
      targets.contiguous().data_ptr<int>(),
      num_classes,
      gamma,
      alpha,
      buff.data_ptr<scalar_t>());
  });

  AT_DISPATCH_FLOATING_TYPES(logits.scalar_type(), "SoftmaxFocalLoss_backward", [&] {
    SoftmaxFocalLossBackward<scalar_t><<<grid2, block, 0, stream>>>(
      d_logits_size,
      P.contiguous().data_ptr<scalar_t>(),
      targets.contiguous().data_ptr<int>(),
      d_losses.contiguous().data_ptr<scalar_t>(),
      buff.data_ptr<scalar_t>(),
      num_classes,
      gamma,
      alpha,
      d_logits.data_ptr<scalar_t>());
  });

  THCudaCheck(cudaGetLastError());
  return d_logits;
}
