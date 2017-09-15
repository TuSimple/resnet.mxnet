# resnet.mxnet

This repository contains a [MXNet](https://github.com/apache/incubator-mxnet) implementation for the [ResNet-v2](https://arxiv.org/abs/1603.05027) and [ResNeXt](https://arxiv.org/abs/1611.05431) algorithms for image classification.

## Requirements and Dependencies

- Install [a modified MXNet](https://github.com/huangzehao/incubator-mxnet-bk) on a machine with CUDA GPU
- Install CUDNN v5 or v6
- Download the [ImageNet](http://image-net.org/download-images) dataset and creat pass through rec (following [tornadomeet's repository](https://github.com/tornadomeet/ResNet#imagenet) but using unchange mode)

## Train and Test

- Modify ```config/cfg.py``` by your setting
- ```python train.py```
- ```python test.py```

## Reference
[1] Kaiming He, et al. "Identity Mappings in Deep Residual Networks" arXiv:1603.05027 (2016).

[2] Xie, Saining, et al. "Aggregated residual transformations for deep neural networks." arXiv:1611.05431 (2016).

[3] Chen, Tianqi, et al. "Mxnet: A flexible and efficient machine learning library for heterogeneous distributed systems." arXiv:1512.01274 (2015).

[4] Torch training code and model provided by facebook, https://github.com/facebookresearch/ResNeXt

[5] MXNet training code provided by tornadomeet, https://github.com/tornadomeet/ResNet
