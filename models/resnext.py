#!/usr/bin/env python
# -*- coding: utf-8 -*-

import chainer
import chainer.functions as F
import chainer.links as L
from chainer.initializers import normal
import collections


class BuildingBlock(chainer.link.Chain):

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

    def __init__(self, n_layer, in_channels, out_channels,
                 cardinality, base_width, widen_factor, stride, initialW=None):
        super(BuildingBlock, self).__init__()
        with self.init_scope():
            self.a = BottleneckA(
                in_channels, out_channels, cardinality, base_width, widen_factor, stride,
                initialW)
            self._forward = ["a"]
            for i in range(n_layer - 1):
                name = 'b{}'.format(i + 1)
                bottleneck = BottleneckB(out_channels, cardinality, base_width, widen_factor, initialW)
                setattr(self, name, bottleneck)
                self._forward.append(name)

    def __call__(self, x):
        for name in self._forward:
            layer = getattr(self, name)
            x = layer(x)
        return x

    @property
    def forward(self):
        return [getattr(self, name) for name in self._forward]


class BottleneckA(chainer.link.Chain):

    """A bottleneck layer that reduces the resolution of the feature map.

    Args:
        in_channels (int): Number of channels of input arrays.
        mid_channels (int): Number of channels of intermediate arrays.
        out_channels (int): Number of channels of output arrays.
        stride (int or tuple of ints): Stride of filter application.
        initialW (4-D array): Initial weight value used in
            the convolutional layers.
    """

    def __init__(self, in_channels, out_channels, cardinality, base_width, widen_factor,
                 stride=2, initialW=None):
        super(BottleneckA, self).__init__()

        width_ratio = out_channels / (widen_factor * 64)
        D = cardinality * int(base_width * width_ratio)
        with self.init_scope():
            self.conv1 = L.Convolution2D(
                in_channels, D, 1, 1, 0, initialW=initialW,
                nobias=True)
            self.bn1 = L.BatchNormalization(D)
            self.conv2 = L.Convolution2D(
                D, D, 3, stride, 1,
                initialW=initialW, nobias=True, group=cardinality)
            self.bn2 = L.BatchNormalization(D)
            self.conv3 = L.Convolution2D(
                D, out_channels, 1, 1, 0, initialW=initialW,
                nobias=True)
            self.bn3 = L.BatchNormalization(out_channels)
            self.conv4 = L.Convolution2D(
                in_channels, out_channels, 1, stride, 0, initialW=initialW,
                nobias=True)
            self.bn4 = L.BatchNormalization(out_channels)

    def __call__(self, x):
        h1 = F.relu(self.bn1(self.conv1(x)))
        h1 = F.relu(self.bn2(self.conv2(h1)))
        h1 = self.bn3(self.conv3(h1))
        h2 = self.bn4(self.conv4(x))
        return F.relu(h1 + h2)


class BottleneckB(chainer.link.Chain):

    """A bottleneck layer that maintains the resolution of the feature map.

    Args:
        in_channels (int): Number of channels of input and output arrays.
        mid_channels (int): Number of channels of intermediate arrays.
        initialW (4-D array): Initial weight value used in
            the convolutional layers.
    """

    def __init__(self, in_channels, cardinality, base_width, widen_factor, initialW=None):
        super(BottleneckB, self).__init__()
        width_ratio = in_channels / (widen_factor * 64)
        D = cardinality * int(base_width * width_ratio)
        with self.init_scope():
            self.conv1 = L.Convolution2D(
                in_channels, D, 1, 1, 0, initialW=initialW,
                nobias=True)
            self.bn1 = L.BatchNormalization(D)
            self.conv2 = L.Convolution2D(
                D, D, 3, 1, 1, initialW=initialW,
                nobias=True, group=cardinality)
            self.bn2 = L.BatchNormalization(D)
            self.conv3 = L.Convolution2D(
                D, in_channels, 1, 1, 0, initialW=initialW,
                nobias=True)
            self.bn3 = L.BatchNormalization(in_channels)

    def __call__(self, x):
        h = F.relu(self.bn1(self.conv1(x)))
        h = F.relu(self.bn2(self.conv2(h)))
        h = self.bn3(self.conv3(h))
        return F.relu(h + x)


class ResNeXt(chainer.Chain):

    def __init__(self, n_layers, n_out, cardinality=8, base_width=64, widen_factor=4,
                 layer_names=None):
        super().__init__()
        kwargs = {'initialW': normal.HeNormal(scale=1.0)}

        if (n_layers - 2) % 9 == 0:
            block = [(n_layers - 2) // 9] * 3
        else:
            raise ValueError(
                'The n_layers argument should be mod({} - 2, 9) == 0,  \
                 but {} was given.'.format(n_layers, n_layers))

        with self.init_scope():
            self.conv1 = L.Convolution2D(None, 64, 3, 1, 1, **kwargs)
            self.bn1 = L.BatchNormalization(64)
            self.res2 = BuildingBlock(block[0], 64, 64 * widen_factor, cardinality, base_width, widen_factor, 1, **kwargs)
            self.res3 = BuildingBlock(block[1], 64 * widen_factor, 128 * widen_factor, cardinality, base_width, widen_factor, 2, **kwargs)
            self.res4 = BuildingBlock(block[2], 128 * widen_factor, 256 * widen_factor, cardinality, base_width, widen_factor, 2, **kwargs)
            self.fc5 = L.Linear(256 * widen_factor, n_out)

        self.functions = collections.OrderedDict([
            ('conv1', [self.conv1, self.bn1, F.relu]),
            ('res2', [self.res2]),
            ('res3', [self.res3]),
            ('res4', [self.res4]),
            ('pool4', [lambda x: F.average_pooling_2d(x, 8, stride=1)]),
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
