# Parallel Grid Pooling
This repository contains the code for the paper Parallel Grid Pooling. 

## Requirements
- Python 3.5+
- Chainer 4.0.0b2
- ChainerCV 0.8.0

## Training
    $ python train.py --dataset [cifar10,cifar100,svhn] --model PreResNet164 --gpus 0

    $ python train_imagenet.py --model ResNet50_fb --gpus 0,1,2,3,4,5,6,7

## Results on ImageNet and Pretrained Models
The error rates shown are 224x224 1-crop test errors.

| Network | #Params | Top-1 error | Top-5 error | Model |
| ------- |--------:| ----------: | ----------: | ----- |
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
