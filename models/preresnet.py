#!/usr/bin/env python
# -*- coding: utf-8 -*-

import chainer
import chainer.functions as F
import chainer.links as L
from chainer.initializers import normal
import collections


class PreBuildingBlock(chainer.Chain):

    """A building block that consists of several Bottleneck layers.

    Args:
        n_layer (int): Number of layers used in the building block.
        in_channels (int): Number of channels of input arrays.
        mid_channels (int): Number of channels of intermediate arrays.
        out_channels (int): Number of channels of output arrays.
        stride (int or tuple of ints): Stride of filter application.
        initialW (4-D array): Initial weight value used in
            the convolutional layers.
    """

    def __init__(self, n_layer, in_channels, mid_channels,
                 out_channels, stride, initialW=None):
        super().__init__()
        with self.init_scope():
            self.a = PreBottleneckA(
                in_channels, mid_channels, out_channels, stride, initialW)
            self.forward = [self.a]
            for i in range(n_layer - 1):
                name = 'b{}'.format(i + 1)
                bottleneck = PreBottleneckB(out_channels, mid_channels,
                                            initialW)
                setattr(self, name, bottleneck)
                self.forward.append(bottleneck)

    def __call__(self, x):
        for l in self.forward:
            x = l(x)
        return x


class PreBottleneckA(chainer.Chain):

    """A bottleneck layer that reduces the resolution of the feature map.

    Args:
        in_channels (int): Number of channels of input arrays.
        mid_channels (int): Number of channels of intermediate arrays.
        out_channels (int): Number of channels of output arrays.
        stride (int or tuple of ints): Stride of filter application.
        initialW (4-D array): Initial weight value used in
            the convolutional layers.
    """

    def __init__(self, in_channels, mid_channels, out_channels,
                 stride=2, initialW=None):
        super().__init__()
        with self.init_scope():
            self.bn1 = L.BatchNormalization(in_channels)
            self.conv1 = L.Convolution2D(
                in_channels, mid_channels, 1, stride, 0, initialW=initialW,
                nobias=True)
            self.bn2 = L.BatchNormalization(mid_channels)
            self.conv2 = L.Convolution2D(
                mid_channels, mid_channels, 3, 1, 1, initialW=initialW,
                nobias=True)
            self.bn3 = L.BatchNormalization(mid_channels)
            self.conv3 = L.Convolution2D(
                mid_channels, out_channels, 1, 1, 0, initialW=initialW,
                nobias=True)
            self.conv4 = L.Convolution2D(
                in_channels, out_channels, 1, stride, 0, initialW=initialW,
                nobias=True)

    def __call__(self, x):
        h0 = F.relu(self.bn1(x))
        h1 = self.conv1(h0)
        h1 = self.conv2(F.relu(self.bn2(h1)))
        h1 = self.conv3(F.relu(self.bn3(h1)))
        h2 = self.conv4(h0)
        return h1 + h2


class PreBottleneckB(chainer.Chain):

    """A bottleneck layer that maintains the resolution of the feature map.

    Args:
        in_channels (int): Number of channels of input and output arrays.
        mid_channels (int): Number of channels of intermediate arrays.
        initialW (4-D array): Initial weight value used in
            the convolutional layers.
    """

    def __init__(self, in_channels, mid_channels, initialW=None):
        super().__init__()
        with self.init_scope():
            self.bn1 = L.BatchNormalization(in_channels)
            self.conv1 = L.Convolution2D(
                in_channels, mid_channels, 1, 1, 0, initialW=initialW,
                nobias=True)
            self.bn2 = L.BatchNormalization(mid_channels)
            self.conv2 = L.Convolution2D(
                mid_channels, mid_channels, 3, 1, 1, initialW=initialW,
                nobias=True)
            self.bn3 = L.BatchNormalization(mid_channels)
            self.conv3 = L.Convolution2D(
                mid_channels, in_channels, 1, 1, 0, initialW=initialW,
                nobias=True)

    def __call__(self, x):
        h = self.conv1(F.relu(self.bn1(x)))
        h = self.conv2(F.relu(self.bn2(h)))
        h = self.conv3(F.relu(self.bn3(h)))
        return h + x


class PreResNet(chainer.Chain):

    def __init__(self, n_layers, n_out, stride=(1, 2, 2), layer_names=None):
        super().__init__()
        kwargs = {'initialW': normal.HeNormal(scale=1.0)}
        if (n_layers - 2) % 3 == 0:
            block = [(n_layers - 2) // 9] * 3
        else:
            raise ValueError(
                'The n_layers argument should be mod({} - 2, 3) == 0,  \
                 but {} was given.'.format(n_layers, n_layers))

        with self.init_scope():
            self.conv1 = L.Convolution2D(3, 16, 3, 1, 1, **kwargs)
            self.res2 = PreBuildingBlock(block[0], 16, 16, 64, stride[0],
                                         **kwargs)
            self.res3 = PreBuildingBlock(block[1], 64, 32, 128, stride[1],
                                         **kwargs)
            self.res4 = PreBuildingBlock(block[2], 128, 64, 256, stride[2],
                                         **kwargs)
            self.bn4 = L.BatchNormalization(256)
            self.fc5 = L.Linear(None, n_out)

        self.functions = collections.OrderedDict([
            ('conv1', [self.conv1]),
            ('res2', [self.res2]),
            ('res3', [self.res3]),
            ('res4', [self.res4]),
            ('pool4', [self.bn4, F.relu, lambda x: F.average(x, axis=(2, 3))]),
            ('fc5', [self.fc5]),
        ])

        if layer_names is None:
            layer_names = list(self.functions.keys())[-1]
        if (not isinstance(layer_names, str) and
                all([isinstance(name, str) for name in layer_names])):
            return_tuple = True
        else:
            return_tuple = False
            layer_names = [layer_names]
        self._return_tuple = return_tuple
        self._layer_names = layer_names

    def __call__(self, x):
        h = x

        activations = dict()
        target_layers = set(self._layer_names)
        for key, funcs in self.functions.items():
            if len(target_layers) == 0:
                break
            for func in funcs:
                h = func(h)
            if key in target_layers:
                activations[key] = h
                target_layers.remove(key)

        if self._return_tuple:
            activations = tuple(
                [activations[name] for name in self._layer_names])
        else:
            activations = list(activations.values())[0]
        return activations

    def extract(self, images, layers=['fc5']):
        self._layer_names = layers
        x = chainer.Variable(self.xp.asarray(images))
        return chainer.cuda.to_cpu(self(x).data)
