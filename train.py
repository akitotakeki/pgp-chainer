#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import os
import sys
import argparse

import chainer
from chainer import training
from chainer.iterators import MultithreadIterator
from chainer.training import extensions, triggers
from chainer.training.updaters import MultiprocessParallelUpdater
# from chainer.dataset import convert
from chainer.datasets import split_dataset_n_random
from chainer.datasets import TransformDataset
import chainercv.transforms as T

from optimizer import CosineAnnealing
import opts

chainer.global_config.autotune = True
chainer.config.type_check = False


if __name__ == '__main__':
    models = opts.model_dict

    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='cifar10',
                        choices=('cifar10', 'cifar100', 'svhn'), help='Dataset')
    parser.add_argument('--batchsize', type=int, default=64)
    parser.add_argument('--model', type=str, default='PreResNet164',
                        choices=tuple(models.keys()))
    parser.add_argument('--epoch', '-e', type=int, default=300,
                        help='Number of sweeps over the dataset \
                        to train (default: 160)')
    parser.add_argument('--gpus', type=str, default='0',
                        help='Set GPU device numbers with comma saparated. '
                        'Default is 0.')
    parser.add_argument('--lr', type=float, default=0.2,
                        help='Initial learning rate (default: 0.2)')
    parser.add_argument('--opt', type=str, default='momentum',
                        choices=('momentum', 'nesterov'),
                        help='Optimizer')
    parser.add_argument('--cosine', default=True, help='Use cosine annealing')
    parser.add_argument('--eta_min', type=float, default=0.002,
                        help='The lower bound of learning rate (default: 0.002)')
    parser.add_argument('--trial', type=str, default=None,
                        help='Trial number')
    parser.add_argument('--output', default='result')
    parser.add_argument('--ow', help='Overwrite', action='store_true')
    parser.add_argument('--resume')
    parser.add_argument('--aug', default=True, help='Standard augmentation')
    parser.add_argument('--nclasses', '-n', type=int, default=10,
                        help='Number of classes')
    args = parser.parse_args()

    # Decide dirname
    _dirname = [args.model]
    _dirname.append('lr' + str(args.lr))
    if args.cosine:
        _dirname.append('cos')
    if not args.opt == 'momentum':
        _dirname.append(args.opt)
    _dirname.append(str(args.epoch) + 'epoch')
    _dirname.append('batchsize' + str(args.batchsize))

    if args.trial:
        _dirname.append('trial' + args.trial)

    dirname = '_'.join(_dirname)
    print('Train: ', dirname)
    output_dir = os.path.join(args.output, args.dataset, dirname)
    if os.path.exists(output_dir) and not args.ow:
        sys.exit('{} already exists. '.format(output_dir) +
                 'Please add option \'--ow True\' if you overwrite.')

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

    # Preprocessing
    # images = convert.concat_examples(train)[0]
    # mean = images.mean(axis=(0, 2, 3))
    # std = images.std(axis=(0, 2, 3))
    # print(mean, std)

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

    train = TransformDataset(
        train, lambda x: transform(x, mean, std, aug=args.aug))
    test = TransformDataset(test, lambda x: transform(x, mean, std))

    # setup model
    model = models[args.model](args)

    args.gpus = list(map(int, args.gpus.split(',')))
    devices = {'main': args.gpus[0]}
    for gid in args.gpus[1:]:
        devices['gpu{}'.format(gid)] = gid

    if len(args.gpus) < 2:
        chainer.cuda.get_device_from_id(args.gpus[0]).use()
        model.to_gpu()

    # Setup an optimizer
    if args.opt == 'nesterov':
        optimizer = chainer.optimizers.NesterovAG()
    else:
        optimizer = chainer.optimizers.MomentumSGD()

    optimizer.setup(model)
    optimizer.add_hook(chainer.optimizer.WeightDecay(1e-4))

    if len(args.gpus) < 2:
        train_iter = chainer.iterators.SerialIterator(train, args.batchsize)
    else:
        train_iters = [
            MultithreadIterator(i, int(args.batchsize / len(devices)),
                                n_threads=4)
            for i in split_dataset_n_random(train, len(devices))]

    test_iter = MultithreadIterator(test, args.batchsize, repeat=False,
                                    shuffle=False, n_threads=4)

    # Set up a trainer
    if len(args.gpus) < 2:
        updater = training.StandardUpdater(train_iter, optimizer,
                                           device=args.gpus[0])
    else:
        updater = MultiprocessParallelUpdater(train_iters, optimizer,
                                              devices=devices)
    trainer = training.Trainer(updater, (args.epoch, 'epoch'),
                               out=output_dir)

    if args.cosine:
        trainer.extend(
            CosineAnnealing('lr', int(args.epoch),
                            len(train) / args.batchsize,
                            eta_min=args.eta_min,
                            init=args.lr))
    else:
        trainer.extend(
            extensions.ExponentialShift('lr', 0.1, init=args.lr),
            trigger=triggers.ManualScheduleTrigger(
                [int(args.epoch * 0.50), int(args.epoch * 0.75)], 'epoch'))

    test_interval = 1, 'epoch'
    snapshot_interval = 10, 'epoch'
    log_interval = 100, 'iteration'

    trainer.extend(extensions.Evaluator(test_iter, model,
                   device=args.gpus[0]), trigger=test_interval)
    trainer.extend(extensions.dump_graph('main/loss'))
    trainer.extend(extensions.snapshot(
        filename='snapshot_epoch_{.updater.epoch}'), trigger=snapshot_interval)
    trainer.extend(extensions.snapshot_object(
        model, 'model_epoch_{.updater.epoch}'), trigger=snapshot_interval)
    trainer.extend(extensions.LogReport(trigger=log_interval))
    trainer.extend(extensions.observe_lr(), trigger=log_interval)

    log_list = ['epoch', 'iteration',
                'main/loss', 'main/accuracy',
                'validation/main/loss', 'validation/main/accuracy', 'lr',
                'elapsed_time']

    trainer.extend(extensions.PrintReport(log_list), trigger=log_interval)
    trainer.extend(extensions.ProgressBar(update_interval=10))

    if args.resume:
        # Resume from a snapshot
        chainer.serializers.load_npz(args.resume, trainer)

    trainer.run()
