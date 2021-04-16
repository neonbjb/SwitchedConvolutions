#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>

using namespace std;

#define CUDA_KERNEL_LOOP(i, n)                                                 \
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < (n);                 \
       i += blockDim.x * gridDim.x)

inline int GET_BLOCKS(const int N, const int num_threads) {
  return (N + num_threads - 1) / num_threads;
}

template <typename scalar_t>
__global__ void switched_conv_cuda_forward_kernel(
  const torch::PackedTensorAccessor32<scalar_t,4,torch::RestrictPtrTraits> x,
  const torch::PackedTensorAccessor32<int,3,torch::RestrictPtrTraits> mask,
  const torch::PackedTensorAccessor32<scalar_t,5,torch::RestrictPtrTraits> kernel,
  torch::PackedTensorAccessor32<scalar_t,4,torch::RestrictPtrTraits> out,
  const int num_kernels,
  const int height,
  const int width,
  const int channels_in,
  const int channels_out,
  const int kernel_size,
  const int stride) {
  // Each kernel has it's own "range of responsibility", which is computed via this macro and stored in "index".
  CUDA_KERNEL_LOOP(index, num_kernels) {
      // Extract variable names we actually can use from 'index'
      const int chan_in_warp = index % 32;
      const int w = (index / 32) % width;
      const int h = (index / 32 / width) % height;
      const int chan_out = (index / 32 / width / height) % channels_out;
      const int b = index / 32 / width / height / channels_out;
      const int mask_location_selection = mask[b][h][w];

      scalar_t local_out = 0;
      const int kernel_shift = kernel_size / 2;
      for(int chan_in = chan_in_warp; chan_in < channels_in; chan_in += 32) {
        for(int kernel_y = -kernel_shift; kernel_y <= kernel_shift; kernel_y++) {
          for(int kernel_x = -kernel_shift; kernel_x <= kernel_shift; kernel_x++) {
            local_out += x[b][h*stride+kernel_y+kernel_shift][w*stride+kernel_x+kernel_shift][chan_in] *
            kernel[mask_location_selection][kernel_y+kernel_shift][kernel_x+kernel_shift][chan_out][chan_in];
          }
        }
      }
      // This equates to a simple reduce_add() for floats.
      for (int offset = 16; offset > 0; offset /= 2) {
        local_out += __shfl_down_sync(0xffffffff, local_out, offset);
      }
      if(threadIdx.x % 32 == 0) {
        out[b][chan_out][h][w] += local_out;
      }
  }
}

torch::Tensor switched_conv_cuda_forward(torch::Tensor x, torch::Tensor mask, torch::Tensor kernel, torch::Tensor bias, int stride) {
  // get shapes
  const int kernel_size = kernel.size(1);
  TORCH_CHECK(kernel_size == kernel.size(2), "kernel must have equivalent kernel size for both dimensions.");
  TORCH_CHECK(kernel_size % 2 == 1, "kernel size must be odd for switched_conv.");
  const int padding = kernel_size / 2;
  const int batch_size = x.size(0);
  const int height = (x.size(1)-padding-padding) / stride;
  const int width = (x.size(2)-padding-padding) / stride;
  const int channels_in = x.size(3);
  TORCH_CHECK(channels_in == kernel.size(4), "input tensor must have the same channels as the input size of the kernel.");
  const int channels_out = kernel.size(3);

  // Configure outputs
  auto out_options = torch::TensorOptions().device(x.device().type(), x.device().index()).dtype(x.dtype());
  auto out = torch::zeros({batch_size, channels_out, height, width}, out_options);

  // Configure device settings
  const int CUDA_NUM_THREADS = 256;
  cudaDeviceSetCacheConfig(cudaFuncCachePreferL1);  // The naive kernel does not use shared memory, so prefer L1 cache.

  // Engage hyperdrive
  int num_kernels = batch_size * channels_out * height * width * 32;
  AT_DISPATCH_FLOATING_TYPES(x.type(), "switched_conv_forward_cuda", ([&] {
    switched_conv_cuda_forward_kernel<scalar_t><<<GET_BLOCKS(num_kernels, CUDA_NUM_THREADS), CUDA_NUM_THREADS>>>(
      x.packed_accessor32<scalar_t,4,torch::RestrictPtrTraits>(),
      mask.packed_accessor32<int,3,torch::RestrictPtrTraits>(),
      kernel.packed_accessor32<scalar_t,5,torch::RestrictPtrTraits>(),
      out.packed_accessor32<scalar_t,4,torch::RestrictPtrTraits>(),
      num_kernels, height, width, channels_in, channels_out, kernel_size, stride
    );
  }));

  cudaError_t err = cudaGetLastError();
  TORCH_CHECK(err == cudaSuccess, cudaGetErrorString(err));
  return out + bias.view({1,channels_out,1,1});
}


template <typename scalar_t>
__global__ void switched_conv_cuda_backward_kernel_grad_input(
  const torch::PackedTensorAccessor32<scalar_t,4,torch::RestrictPtrTraits> x,
  const torch::PackedTensorAccessor32<scalar_t,4,torch::RestrictPtrTraits> x_grad,
  const torch::PackedTensorAccessor32<int,3,torch::RestrictPtrTraits> mask,
  const torch::PackedTensorAccessor32<scalar_t,5,torch::RestrictPtrTraits> kernel,
  const torch::PackedTensorAccessor32<scalar_t,1,torch::RestrictPtrTraits> bias,
  torch::PackedTensorAccessor32<scalar_t,5,torch::RestrictPtrTraits> grad,
  const int num_kernels,
  const int height,
  const int width,
  const int channels_in,
  const int channels_out,
  const int kernel_size,
  const int stride) {
  CUDA_KERNEL_LOOP(index, num_kernels) {
      const int chan_in = index % channels_in;
      const int w = (index / channels_in) % width;
      const int h = (index / channels_in / width) % height;
      const int b = index / channels_in / width / height;
      const int mask_location_selection = mask[b][h][w];

      // Perform the computation: iterate along the channels_in dimension and apply the kernel.
      const int kernel_shift = kernel_size / 2;
      for(int kernel_y = -kernel_shift; kernel_y <= kernel_shift; kernel_y++) {
        for(int kernel_x = -kernel_shift; kernel_x <= kernel_shift; kernel_x++) {
          float grad_acc = 0;
          for(int chan_out = 0; chan_out < channels_out; chan_out++) {
            // TODO: Explore shifting x_grad into shared memory to improve access.
            grad_acc += x_grad[b][chan_out][h][w] * kernel[mask_location_selection][kernel_y+kernel_shift][kernel_x+kernel_shift][chan_out][chan_in];
          }
          grad[b][(kernel_y+kernel_shift)*kernel_size+kernel_x+kernel_shift][h*stride+kernel_y+kernel_shift][w*stride+kernel_x+kernel_shift][chan_in] += grad_acc;
        }
      }
  }
}

template <typename scalar_t>
__global__ void switched_conv_cuda_backward_kernel_grad_kernel(
  const torch::PackedTensorAccessor32<scalar_t,4,torch::RestrictPtrTraits> x,
  const torch::PackedTensorAccessor32<scalar_t,4,torch::RestrictPtrTraits> x_grad,
  const torch::PackedTensorAccessor32<int,3,torch::RestrictPtrTraits> mask,
  torch::PackedTensorAccessor32<scalar_t,6,torch::RestrictPtrTraits> gradKernel,
  const int num_kernels,
  const int height,
  const int width,
  const int channels_in,
  const int channels_out,
  const int kernel_size,
  const int batch_size,
  const int breadth,
  const int stride) {
  CUDA_KERNEL_LOOP(index, num_kernels) {
      // Extract variable names we actually can use from 'index'
      const int kernel_shift = kernel_size / 2;
      const int kernel_x = index % kernel_size - kernel_shift;
      const int kernel_y = (index / kernel_size) % kernel_size - kernel_shift;
      const int chan_in = (index / kernel_size / kernel_size) % channels_in;
      const int chan_out = (index / kernel_size / kernel_size / channels_in) % channels_out;
      const int b = index / kernel_size / kernel_size / channels_in / channels_out;

      extern __shared__ float kern_out[];
      for(int i = 0; i < breadth; i++) {
        kern_out[i*blockDim.x+threadIdx.x] = 0;
      }
      __syncwarp();  // Each thread uses its own shared memory - it is not actually "shared". __syncwarp() mainly helps threads achieved coalesced memory access.

      // Perform the computation: iterate along the channels_in dimension and apply the kernel.
      for(int h = 0; h < height; h++) {
        for(int w = 0; w < width; w++) {
          const int mask_location_selection = mask[b][h][w];
          kern_out[mask_location_selection * blockDim.x + threadIdx.x] += x_grad[b][chan_out][h][w] *
              // This kernel assumes that {x} is padded on all sides.
              x[b][chan_in][h*stride+kernel_y+kernel_shift][w*stride+kernel_x+kernel_shift];
        }
      }

      for(int i = 0; i < breadth; i++) {
        gradKernel[b][chan_out][chan_in][i][kernel_y+kernel_shift][kernel_x+kernel_shift] += kern_out[i * blockDim.x + threadIdx.x];
      }
  }
}


vector<torch::Tensor> switched_conv_cuda_backward(torch::Tensor x, torch::Tensor x_grad, torch::Tensor mask, torch::Tensor kernel, torch::Tensor bias, int stride) {
  // get shapes
  const int kernel_size = kernel.size(3);
  TORCH_CHECK(kernel_size == kernel.size(4), "kernel must have equivalent kernel size for last two dimensions.");
  TORCH_CHECK(kernel_size % 2 == 1, "kernel size must be odd for switched_conv.");
  const int padding = kernel_size / 2;
  const int batch_size = x.size(0);
  TORCH_CHECK(batch_size == x_grad.size(0), "x_grad batch dim does not match.");
  TORCH_CHECK(batch_size == mask.size(0), "mask batch dim does not match.");
  const int channels_in = x.size(1);
  TORCH_CHECK(channels_in == kernel.size(1), "input tensor must have the same channels as the input size of the kernel.");
  const int height = (x.size(2)-padding-padding) / stride;
  const int width = (x.size(3)-padding-padding) / stride;
  TORCH_CHECK(height == mask.size(1), "mask width does not match expected output width.");
  TORCH_CHECK(width == mask.size(2), "mask height does not match expected output height.");
  TORCH_CHECK(height == x_grad.size(2), "gradient height does not match input height.");
  TORCH_CHECK(width == x_grad.size(3), "gradient width does not match input width.");
  const int channels_out = kernel.size(0);
  const int breadth = kernel.size(2);

  // Configure outputs
  auto options = torch::TensorOptions().device(x.device().type(), x.device().index()).dtype(x.dtype());
  // To prevent memory access conflicts, x_grad is expanded across the entire kernel -> each kernel index gets it's own entry into x_grad. The result is summed out. Note: there is an algorithmic change that can improve efficiency if stride>1
  auto grad = torch::zeros({batch_size, kernel_size*kernel_size, x.size(2), x.size(3), channels_in}, options);
  auto gradKernel = torch::zeros({batch_size, channels_out, channels_in, breadth, kernel_size, kernel_size}, options);

  // Configure device settings
  const int CUDA_NUM_THREADS = 256;
  cudaDeviceSetCacheConfig(cudaFuncCachePreferL1);  // The naive kernel does not use shared memory, so prefer L1 cache.

  int num_kernels = batch_size * height * width * channels_in;
  auto kern_permute = kernel.clone().permute({2,3,4,0,1}).contiguous();
  AT_DISPATCH_FLOATING_TYPES(x.type(), "switched_conv_backward_cuda_grad_input", ([&] {
    switched_conv_cuda_backward_kernel_grad_input<scalar_t><<<GET_BLOCKS(num_kernels, CUDA_NUM_THREADS), CUDA_NUM_THREADS>>>(
      x.packed_accessor32<scalar_t,4,torch::RestrictPtrTraits>(),
      x_grad.packed_accessor32<scalar_t,4,torch::RestrictPtrTraits>(),
      mask.packed_accessor32<int,3,torch::RestrictPtrTraits>(),
      kern_permute.packed_accessor32<scalar_t,5,torch::RestrictPtrTraits>(),
      bias.packed_accessor32<scalar_t,1,torch::RestrictPtrTraits>(),
      grad.packed_accessor32<scalar_t,5,torch::RestrictPtrTraits>(),
      num_kernels, height, width, channels_in, channels_out, kernel_size, stride
    );
  }));

  num_kernels = batch_size * kernel_size * kernel_size * channels_in * channels_out;
  AT_DISPATCH_FLOATING_TYPES(kernel.type(), "switched_conv_backward_cuda_grad_kernel", ([&] {
    switched_conv_cuda_backward_kernel_grad_kernel<scalar_t><<<GET_BLOCKS(num_kernels, CUDA_NUM_THREADS), CUDA_NUM_THREADS, breadth * CUDA_NUM_THREADS * sizeof(scalar_t)>>>(
      x.packed_accessor32<scalar_t,4,torch::RestrictPtrTraits>(),
      x_grad.packed_accessor32<scalar_t,4,torch::RestrictPtrTraits>(),
      mask.packed_accessor32<int,3,torch::RestrictPtrTraits>(),
      gradKernel.packed_accessor32<scalar_t,6,torch::RestrictPtrTraits>(),
      num_kernels, height, width, channels_in, channels_out, kernel_size, batch_size, breadth, stride
    );
  }));

  // Bias grad is simply the sum of the gradients across everything except the chan_out dimension.
  auto gradBias = x_grad.sum({0,2,3});

  // For proper concurrent access patterns, some tensors were expanded. Sum them now.
  grad = grad.sum(1).permute({0,3,1,2}).contiguous();
  gradKernel = gradKernel.sum(0);
  return {grad, gradKernel, gradBias};
}


torch::Tensor cuda_bench_conv_grad_input(torch::Tensor x, torch::Tensor x_grad, torch::Tensor mask, torch::Tensor kernel, torch::Tensor bias, int stride) {
  // get shapes
  const int kernel_size = kernel.size(3);
  TORCH_CHECK(kernel_size == kernel.size(4), "kernel must have equivalent kernel size for last two dimensions.");
  TORCH_CHECK(kernel_size % 2 == 1, "kernel size must be odd for switched_conv.");
  const int padding = kernel_size / 2;
  const int batch_size = x.size(0);
  TORCH_CHECK(batch_size == x_grad.size(0), "x_grad batch dim does not match.");
  TORCH_CHECK(batch_size == mask.size(0), "mask batch dim does not match.");
  const int channels_in = x.size(1);
  TORCH_CHECK(channels_in == kernel.size(1), "input tensor must have the same channels as the input size of the kernel.");
  const int height = (x.size(2)-padding-padding) / stride;
  const int width = (x.size(2)-padding-padding) / stride;
  TORCH_CHECK(height == mask.size(1), "mask width does not match expected output width.");
  TORCH_CHECK(width == mask.size(2), "mask height does not match expected output height.");
  TORCH_CHECK(height == x_grad.size(2), "gradient height does not match input height.");
  TORCH_CHECK(width == x_grad.size(3), "gradient width does not match input width.");
  const int channels_out = kernel.size(0);
  const int breadth = kernel.size(2);

  // Configure outputs
  auto options = torch::TensorOptions().device(x.device().type(), x.device().index()).dtype(x.dtype());
  // To prevent memory access conflicts, x_grad is expanded across the entire kernel -> each kernel index gets it's own entry into x_grad. The result is summed out. Note: there is an algorithmic change that can improve efficiency if stride>1
  auto grad = torch::zeros({batch_size, kernel_size*kernel_size, x.size(2), x.size(3), channels_in}, options);

  // Configure device settings
  const int CUDA_NUM_THREADS = 256;
  cudaDeviceSetCacheConfig(cudaFuncCachePreferL1);  // The naive kernel does not use shared memory, so prefer L1 cache.

  int num_kernels = batch_size * height * width * channels_in;
  auto kern_permute = kernel.clone().permute({2,3,4,0,1}).contiguous();
  AT_DISPATCH_FLOATING_TYPES(x.type(), "switched_conv_backward_cuda_grad_input", ([&] {
    switched_conv_cuda_backward_kernel_grad_input<scalar_t><<<GET_BLOCKS(num_kernels, CUDA_NUM_THREADS), CUDA_NUM_THREADS>>>(
      x.packed_accessor32<scalar_t,4,torch::RestrictPtrTraits>(),
      x_grad.packed_accessor32<scalar_t,4,torch::RestrictPtrTraits>(),
      mask.packed_accessor32<int,3,torch::RestrictPtrTraits>(),
      kern_permute.packed_accessor32<scalar_t,5,torch::RestrictPtrTraits>(),
      bias.packed_accessor32<scalar_t,1,torch::RestrictPtrTraits>(),
      grad.packed_accessor32<scalar_t,5,torch::RestrictPtrTraits>(),
      num_kernels, height, width, channels_in, channels_out, kernel_size, stride
    );
  }));

  // For proper concurrent access patterns, some tensors were expanded. Sum them now.
  grad = grad.sum(1).permute({0,3,1,2}).contiguous();
  return grad;
}


torch::Tensor cuda_bench_conv_grad_kernel(torch::Tensor x, torch::Tensor x_grad, torch::Tensor mask, torch::Tensor kernel, torch::Tensor bias, int stride) {
  // get shapes
  const int kernel_size = kernel.size(3);
  TORCH_CHECK(kernel_size == kernel.size(4), "kernel must have equivalent kernel size for last two dimensions.");
  TORCH_CHECK(kernel_size % 2 == 1, "kernel size must be odd for switched_conv.");
  const int padding = kernel_size / 2;
  const int batch_size = x.size(0);
  TORCH_CHECK(batch_size == x_grad.size(0), "x_grad batch dim does not match.");
  TORCH_CHECK(batch_size == mask.size(0), "mask batch dim does not match.");
  const int channels_in = x.size(1);
  TORCH_CHECK(channels_in == kernel.size(1), "input tensor must have the same channels as the input size of the kernel.");
  const int height = (x.size(2)-padding-padding) / stride;
  const int width = (x.size(2)-padding-padding) / stride;
  TORCH_CHECK(height == mask.size(1), "mask width does not match expected output width.");
  TORCH_CHECK(width == mask.size(2), "mask height does not match expected output height.");
  TORCH_CHECK(height == x_grad.size(2), "gradient height does not match input height.");
  TORCH_CHECK(width == x_grad.size(3), "gradient width does not match input width.");
  const int channels_out = kernel.size(0);
  const int breadth = kernel.size(2);

  // Configure outputs
  auto options = torch::TensorOptions().device(x.device().type(), x.device().index()).dtype(x.dtype());
  // To prevent memory access conflicts, x_grad is expanded across the entire kernel -> each kernel index gets it's own entry into x_grad. The result is summed out. Note: there is an algorithmic change that can improve efficiency if stride>1
  auto gradKernel = torch::zeros({batch_size, channels_out, channels_in, breadth, kernel_size, kernel_size}, options);

  // Configure device settings
  const int CUDA_NUM_THREADS = 256;
  cudaDeviceSetCacheConfig(cudaFuncCachePreferL1);  // The naive kernel does not use shared memory, so prefer L1 cache.

  int num_kernels = batch_size * kernel_size * kernel_size * channels_in * channels_out;
  AT_DISPATCH_FLOATING_TYPES(kernel.type(), "switched_conv_backward_cuda_grad_kernel", ([&] {
    switched_conv_cuda_backward_kernel_grad_kernel<scalar_t><<<GET_BLOCKS(num_kernels, CUDA_NUM_THREADS), CUDA_NUM_THREADS, breadth * CUDA_NUM_THREADS * sizeof(scalar_t)>>>(
      x.packed_accessor32<scalar_t,4,torch::RestrictPtrTraits>(),
      x_grad.packed_accessor32<scalar_t,4,torch::RestrictPtrTraits>(),
      mask.packed_accessor32<int,3,torch::RestrictPtrTraits>(),
      gradKernel.packed_accessor32<scalar_t,6,torch::RestrictPtrTraits>(),
      num_kernels, height, width, channels_in, channels_out, kernel_size, batch_size, breadth, stride
    );
  }));

  gradKernel = gradKernel.sum(0);
  return gradKernel;
}