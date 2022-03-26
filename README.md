# Switched Convolutions
Switched convolutions are convolutions with multiple kernels that are selected via a MoE mechanism.
They can be dropped in to replace most "normal" convolutions but multiplicatively increase parameter
count without increasing depth. They are particularly useful in generative networks.

**Important note**: After much experimentation, I have found that switched convolutions are only beneficial in select environments. I would not recommend using this repo except for experimental purposes. I believe the concept is sound, but it is missing "something" that would help it work correctly. Or there's a bug.

See the blog post [here](https://nonint.com/2021/04/15/switched-convolutions-spatial-moe-for-convolutions/) for more information.


# License

This code is under the Apache 2.0 license. I ask that you cite my name or this page if you use this code or the concepts
therein in your own projects.
