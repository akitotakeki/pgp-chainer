# Parallel Grid Pooling for Data Augmentation
This repository contains the code for the paper [Parallel Grid Pooling for Data Augmentation](https://arxiv.org/abs/1803.11370). 

## Requirements
- Python 3.5+
- Chainer 4.0.0b2+
- CuPy 4.0.0b2+
- ChainerCV 0.8.0+

## Training
To train PreResNet-164 on CIFAR-10 dataset with single-GPU:

    $ python train.py --dataset cifar10 --model PreResNet164 --gpus 0
To train ResNet-50 on ImageNet dataset with multi-GPU:

    $ python train_imagenet.py --model ResNet50_fb --gpus 0,1,2,3,4,5,6,7

## Results on CIFAR-10
Test errors (%)

| Network | #Params | Base | DConv | PGP |
| :------ |--------:| ---: | ----: | --: |
| PreResNet-164          | 1.7M | 4.71 | 4.15 | **3.77** |
| All-CNN                | 1.4M | 8.42 | 8.68 | **7.17** |
| WideResNet-28-10       | 36.5M | 3.44 | 3.88 | **3.13** |
| ResNeXt-29 (8x64d)     | 34.4M | 3.86 | 3.87 | **3.22** |
| PyramidNet-164 (α=28)  | 1.7M | 3.91 | 3.72 | **3.38** |
| DenseNet-BC-100 (k=12) | 0.8M | 4.60 | 4.35 | **4.11** |

### Weight Transfer
Test errors (%) (Test-time data augmentation)

| Network | #Params | Base | PGP |
| :------ |--------:| ---: | --: |
| PreResNet-164          | 1.7M | 4.71 | **4.56** |
| All-CNN                | 1.4M | **8.42** | 9.03 |
| WideResNet-28-10       | 36.5M | 3.44 | **3.39** |
| ResNeXt-29 (8x64d)     | 34.4M | **3.86** | 4.01 |
| PyramidNet-164 (α=28)  | 1.7M | 3.91 | **3.82** |
| DenseNet-BC-100 (k=12) | 0.8M | 4.60 | **4.53** |

Test errors (%) (Training-time data augmentation)

| Network | #Params | Base | DConv | PGP |
| :------ |--------:| ---: | ----: | --: |
| PreResNet-164          | 1.7M | 4.71 | 7.30 | **4.08** |
| All-CNN                | 1.4M | 8.42 | 38.77 | **7.30** |
| WideResNet-28-10       | 36.5M | 3.44 | 7.90 | **3.30** |
| ResNeXt-29 (8x64d)     | 34.4M | 3.86 | 16.91 | **3.36** |
| PyramidNet-164 (α=28)  | 1.7M | 3.91 | 6.82 | **3.55** |
| DenseNet-BC-100 (k=12) | 0.8M | 4.60 | 7.03 | **4.36** |

## Results on ImageNet and Pretrained Models
The error rates (%) shown are 224x224 1-crop test errors.

| Network | #Params | Top-1 error | Top-5 error | Model |
| :------ |--------:| ----------: | ----------: | ----- |
| ResNet-50  (Train: Base, Test: Base)  | 25.6M | 23.69       | 7.00        | [Download (91.1MB)](https://www.hal.t.u-tokyo.ac.jp/~takeki/pgp-chainer/ResNet50_fb) |
| ResNet-50  (Train: DConv, Test: DConv)  | 25.6M | 22.47       | **6.27**    | [Download (91.1MB)](https://www.hal.t.u-tokyo.ac.jp/~takeki/pgp-chainer/ResNet50_fb_DConv) |
| ResNet-50  (Train: PGP,   Test: PGP)    | 25.6M | **22.40**   | 6.30        | [Download (91.1MB)](https://www.hal.t.u-tokyo.ac.jp/~takeki/pgp-chainer/ResNet50_fb_PGP) |
| ResNet-50  (Train: Base,  Test: PGP)    | 25.6M | 23.32       | 6.85        |-|
| ResNet-50  (Train: DConv, Test: Base)   | 25.6M | 31.44       | 11.40       |-|
| ResNet-50  (Train: PGP,   Test: Base)   | 25.6M | 23.01       | 6.66        |-|
| ResNet-101  (Train: Base,  Test: Base)   | 44.5M | 22.49       | 6.38        | [Download (160MB)](https://www.hal.t.u-tokyo.ac.jp/~takeki/pgp-chainer/ResNet101_fb) |
| ResNet-101  (Train: DConv, Test: DConv)  | 44.5M | **21.26**   | **5.61**    | [Download (160MB)](https://www.hal.t.u-tokyo.ac.jp/~takeki/pgp-chainer/ResNet101_fb_DConv) |
| ResNet-101  (Train: PGP,   Test: PGP)    | 44.5M | 21.34       | 5.65        | [Download (160MB)](https://www.hal.t.u-tokyo.ac.jp/~takeki/pgp-chainer/ResNet101_fb_PGP) |
| ResNet-101  (Train: Base,  Test: PGP)    | 44.5M | 22.13       | 6.21        |-|
| ResNet-101  (Train: DConv, Test: Base)   | 44.5M | 25.63       | 8.01        |-|
| ResNet-101  (Train: PGP,   Test: Base)   | 44.5M | 21.80       | 5.95        |-|

## Citation
