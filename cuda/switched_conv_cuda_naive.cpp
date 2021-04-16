#include <torch/extension.h>
#include <vector>

using namespace std;

torch::Tensor switched_conv_cuda_forward(torch::Tensor x, torch::Tensor mask, torch::Tensor kernel, torch::Tensor bias, int stride);
vector<torch::Tensor> switched_conv_cuda_backward(torch::Tensor x, torch::Tensor x_grad, torch::Tensor mask, torch::Tensor kernel, torch::Tensor bias, int stride);
torch::Tensor cuda_bench_conv_grad_input(torch::Tensor x, torch::Tensor x_grad, torch::Tensor mask, torch::Tensor kernel, torch::Tensor bias, int stride);
torch::Tensor cuda_bench_conv_grad_kernel(torch::Tensor x, torch::Tensor x_grad, torch::Tensor mask, torch::Tensor kernel, torch::Tensor bias, int stride);

#define CHECK_CUDA(x) TORCH_CHECK(x.type().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

torch::Tensor switched_conv_forward(torch::Tensor x, torch::Tensor mask, torch::Tensor kernel, torch::Tensor bias, int stride) {
  CHECK_INPUT(x);
  CHECK_INPUT(mask);
  CHECK_INPUT(kernel);
  CHECK_INPUT(bias);
  x = x.permute({0,2,3,1}).contiguous();
  kernel = kernel.permute({2,3,4,0,1}).contiguous();
  return switched_conv_cuda_forward(x, mask, kernel, bias, stride);
}

vector<torch::Tensor> switched_conv_backward(torch::Tensor x, torch::Tensor x_grad, torch::Tensor mask, torch::Tensor kernel, torch::Tensor bias, int stride) {
  CHECK_INPUT(x);
  CHECK_INPUT(x_grad);
  CHECK_INPUT(mask);
  CHECK_INPUT(kernel);
  CHECK_INPUT(bias);
  return switched_conv_cuda_backward(x, x_grad, mask, kernel, bias, stride);
}

torch::Tensor bench_conv_grad_input(torch::Tensor x, torch::Tensor x_grad, torch::Tensor mask, torch::Tensor kernel, torch::Tensor bias, int stride) {
  return cuda_bench_conv_grad_input(x, x_grad, mask, kernel, bias, stride);
}

torch::Tensor bench_conv_grad_kernel(torch::Tensor x, torch::Tensor x_grad, torch::Tensor mask, torch::Tensor kernel, torch::Tensor bias, int stride) {
  return cuda_bench_conv_grad_kernel(x, x_grad, mask, kernel, bias, stride);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &switched_conv_forward, "switched_conv_forward (CUDA)");
  m.def("backward", &switched_conv_backward, "switched_conv_backward (CUDA)");
  m.def("bench_grad_input", &bench_conv_grad_input, "bench_grad_input");
  m.def("bench_grad_kernel", &bench_conv_grad_kernel, "bench_grad_kernel");
}