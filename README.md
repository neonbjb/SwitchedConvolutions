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

The idea is that some types of networks need this form of fine attention across computational blocks. In particular,
I designed this for GAN generators, which are expected to infer underlying style from many possible choices.

This implementation is in Pytorch. It is very much still in test. test.py works great for a basic proof of concept, and 
will output a set of tensorboard logs which should give a great depiction of what is actually happening with the switching 
mechanism. However, ablation on this simple test network shows that the switching provides little to no performace
gain. I expect it will be more relevant in generators and larger networks. I will update this repo with results.


# License

This code is under the Apache 2.0 license. I ask that you cite my name or this page if you use this code or the concepts
therein in your own projects.
