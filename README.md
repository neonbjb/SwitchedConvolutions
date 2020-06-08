# Switched Convolutions
Switched convolutions are trainable convolutions that operate in a similar manner to attention mechanisms. They work by:

1. Performing processing on an input image using convolution blocks.
1. Dropping the filter count of the convolution to the number of input blocks that will be provided to the switch.
1. Performing a softmax across the filters from the last step.
1. Performing a matmul across the softmax output and the input blocks. The end result is a single output that is 
   aggregated from values of the input blocks according to a learned attention.
   
This is inspired by [Dynamic Convolutions](https://arxiv.org/pdf/1912.03458.pdf). The difference is that the attention
mechanism in Dynamic Convolutions selects the computational blocks that will be used across an entire image. Switched
convolutions selects the computational blocks that will be used for each pixel. An obvious caveat is that none of the
computational savings that Dynamic Convolutions expresses is present here: switched convolutions will make your model
significantly more complex both computationally and in memory usage.

This implementation is very much still in test. I designed it for usage in the generators of GANs, where the generator
is expected to be able to create hundreds of different styles. I will update this repo with results.
