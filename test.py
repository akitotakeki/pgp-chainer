#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import argparse
import chainer
from chainer.datasets import TransformDataset
from chainercv.utils import apply_to_iterator
import chainercv.transforms as T
from chainercv.utils import ProgressHook
import opts


if __name__ == '__main__':
    models = opts.model_dict

    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='cifar10',
                        choices=('cifar10', 'cifar100', 'svhn'), help='Dataset')
    parser.add_argument('--testbatch', type=int, default=128)
    parser.add_argument('--model', type=str, default='PreResNet164',
                        choices=tuple(models.keys()))
    parser.add_argument('--gpu', type=int, default=-1)
    parser.add_argument('--weight', type=str, default=None,
                        help='trained weight')
    parser.add_argument('--nclasses', '-n', type=int, default=10,
                        help='Number of classes')
    args = parser.parse_args()

    print('Test: ', args.model)

    # Load dataset
    if args.dataset == 'cifar10':
        train, test = chainer.datasets.get_cifar10()
        args.nclasses = 10
        mean = np.array((0.491401, 0.4821591, 0.44653094), dtype=np.float32)
        std = np.array((0.2470328, 0.24348424, 0.26158753), dtype=np.float32)
    elif args.dataset == 'cifar100':
        train, test = chainer.datasets.get_cifar100()
        args.nclasses = 100
        mean = np.array((0.5070759, 0.48655054, 0.44091946), dtype=np.float32)
        std = np.array((0.267334, 0.25643876, 0.2761503), dtype=np.float32)
    elif args.dataset == 'svhn':
        train, test, extra = chainer.datasets.get_svhn(add_extra=True)
        train = chainer.datasets.ConcatenatedDataset(train, extra)
        args.nclasses = 10
        mean = np.array((0.43091714, 0.43023905, 0.44634134), dtype=np.float32)
        std = np.array((0.19652805, 0.19832036, 0.19942199), dtype=np.float32)

    def transform(in_data, mean, std, aug=False):
        img, label = in_data
        img = img.copy()

        img -= mean[:, None, None]
        img /= std[:, None, None]

        if aug:
            img = T.resize_contain(img, (40, 40))
            img = T.random_crop(img, (32, 32))
            img = T.random_flip(img, x_random=True)

        return img, label

    test = TransformDataset(test, lambda x: transform(x, mean, std))
    test_iter = chainer.iterators.SerialIterator(test, args.testbatch,
                                                 repeat=False, shuffle=False)

    model = models[args.model](args)

    if args.gpu >= 0:
        chainer.cuda.get_device(args.gpu).use()
        model.to_gpu()

    chainer.serializers.load_npz(args.weight, model)

    chainer.config.train = False
    imgs, pred_values, gt_values = apply_to_iterator(
        model.predictor.extract, test_iter, hook=ProgressHook(len(test)))

    del imgs

    features, = pred_values
    labels, = gt_values
    cnt = 0
    for f, l in zip(features, labels):
        if np.argmax(f) == l:
            cnt += 1
    print('Accuracy: ', cnt / len(test))
