#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys
import argparse
import PIL
import numpy as np

import chainer
from chainer import training
from chainer.iterators import MultiprocessIterator
from chainer.training import extensions, triggers
from chainer.datasets import TransformDataset
from optimizer import CosineAnnealing
import chainercv.transforms as T
import chainermn

from transforms import color_jitter, random_sized_crop
import opts_imagenet

chainer.global_config.autotune = True
chainer.config.type_check = False


if __name__ == '__main__':
    models = opts_imagenet.model_dict

    parser = argparse.ArgumentParser()
    parser.add_argument('--batchsize', type=int, default=256)
    parser.add_argument('--test_batchsize', type=int, default=256)
    parser.add_argument('--model', type=str, default='ResNet50_fb',
                        choices=tuple(models.keys()))
    parser.add_argument('--epoch', '-e', type=int, default=90,
                        help='Number of sweeps over the dataset \
                        to train (default: 90)')
    parser.add_argument('--lr', type=float, default=0.1,
                        help='Initial learning rate (default: 0.1)')
    parser.add_argument('--wd', type=str, default='1e-4',
                        help='Weight decay rate (default: 1e-4)')
    parser.add_argument('--opt', type=str, default='momentum',
                        help='Optimizer')
    parser.add_argument('--cosine', default=True, help='Use cosine annealing')
    parser.add_argument('--eta_min', type=float, default=0.0001,
                        help='The lower bound of learning rate (default: 0.0001)')
    parser.add_argument('--trial', type=str, default=None,
                        help='Trial number')
    parser.add_argument('--output', default='result_imagenet')
    parser.add_argument('--ow', help='Overwrite', action='store_true')
    parser.add_argument('--resume')
    parser.add_argument('--nclasses', '-n', type=int, default=1000,
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
    output_dir = os.path.join(args.output, dirname)
    if os.path.exists(output_dir) and not args.ow:
        sys.exit('{} already exists. '.format(output_dir) +
                 'Please add option \'--ow True\' if you overwrite.')

    # Load ImageNet dataset
    train = chainer.datasets.LabeledImageDataset(
        './imagenet_lists/train.txt',
        root='/raid/shared/ILSVRC2012/train')

    test = chainer.datasets.LabeledImageDataset(
        './imagenet_lists/val.txt',
        root='/raid/shared/ILSVRC2012/val')

    # Preprocessing
    mean = np.array((0.485, 0.456, 0.406), dtype=np.float32)
    std = np.array((0.229, 0.224, 0.225), dtype=np.float32)

    def transform(in_data):
        img, label = in_data
        if img.shape[0] == 1:
            img = np.array([img[0], img[0], img[0]])
        elif img.shape[0] == 4:
            img = np.array([img[0], img[1], img[2]])
        img /= 255

        img = random_sized_crop(img)
        img = T.resize(img, (224, 224), interpolation=PIL.Image.BICUBIC)
        img = color_jitter(img)
        img = T.pca_lighting(img, 0.1)

        img = (img - mean[:, None, None]) / std[:, None, None]

        img = T.random_flip(img, x_random=True)
        return img, label

    def transform_test(in_data):
        img, label = in_data
        if img.shape[0] == 1:
            img = np.array([img[0], img[0], img[0]])
        elif img.shape[0] == 4:
            img = np.array([img[0], img[1], img[2]])
        img /= 255

        img = T.scale(img, 256, interpolation=PIL.Image.BICUBIC)
        img = T.center_crop(img, (224, 224))

        img = (img - mean[:, None, None]) / std[:, None, None]

        return img, label

    comm = chainermn.create_communicator('pure_nccl')
    device = comm.intra_rank

    if comm.rank == 0:
        train = TransformDataset(train, transform)
        test = TransformDataset(test, transform_test)
    else:
        train, test = None, None

    # setup model
    model = models[args.model](args)
    chainer.cuda.get_device_from_id(device).use()
    model.to_gpu()

    # Setup an optimizer
    if args.opt == 'nesterov':
        optimizer = chainermn.create_multi_node_optimizer(chainer.optimizers.NesterovAG(), comm)
    else:
        optimizer = chainermn.create_multi_node_optimizer(chainer.optimizers.MomentumSGD(), comm)

    optimizer.setup(model)
    optimizer.add_hook(chainer.optimizer.WeightDecay(float(args.wd)))

    train = chainermn.scatter_dataset(train, comm, shuffle=True)
    test = chainermn.scatter_dataset(test, comm)

    train_iter = MultiprocessIterator(train, args.batchsize // comm.size, n_processes=8)
    test_iter = MultiprocessIterator(test, args.test_batchsize // comm.size, repeat=False,
                                     shuffle=False, n_processes=8)

    updater = training.StandardUpdater(train_iter, optimizer, device=device)
    trainer = training.Trainer(updater, (args.epoch, 'epoch'),
                               out=output_dir)

    if args.cosine:
        trainer.extend(
            CosineAnnealing('lr', int(args.epoch),
                            len(train) / (args.batchsize // comm.size), eta_min=args.eta_min,
                            init=args.lr))
    else:
        mst_epochs = [30, 60, 90]
        trainer.extend(
            extensions.ExponentialShift('lr', 0.1, init=args.lr),
            trigger=triggers.ManualScheduleTrigger(mst_epochs, 'epoch'))

    test_interval = 1, 'epoch'
    snapshot_interval = 10, 'epoch'
    log_interval = 10, 'iteration'

    checkpointer = chainermn.create_multi_node_checkpointer(name='pgp-chainer', comm=comm)
    checkpointer.maybe_load(trainer, optimizer)
    trainer.extend(checkpointer, trigger=test_interval)

    evaluater = extensions.Evaluator(test_iter, model, device=device)
    evaluater = chainermn.create_multi_node_evaluator(evaluater, comm)
    trainer.extend(evaluater, trigger=test_interval)

    if comm.rank == 0:
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
