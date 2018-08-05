#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import chainer
import chainer.functions as F
import chainer.links as L
from chainer.initializers import normal
import collections


class BuildingPyramidBlock(chainer.Chain):

    def __init__(self, n_layer, in_ch, tmp_ch, add_ch, stride=1,
                 initialW=None):
        super().__init__()
        with self.init_scope():
            tmp_ch += add_ch
            self.a = PyramidBottleneck(in_ch, round(tmp_ch),
                                       4 * round(tmp_ch), stride, initialW)
            self.forward = [self.a]
            in_ch = 4 * round(tmp_ch)
            for i in range(n_layer - 1):
                tmp_ch += add_ch
                name = 'b{}'.format(i + 1)
                bottleneck = PyramidBottleneck(in_ch, round(tmp_ch),
                                               4 * round(tmp_ch),
                                               1, initialW)
                setattr(self, name, bottleneck)
                self.forward.append(bottleneck)
                in_ch = 4 * round(tmp_ch)

    def __call__(self, x):
        for l in self.forward:
            x = l(x)
        return x


class PyramidBottleneck(chainer.Chain):

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
                 stride=1, initialW=None):
        super().__init__()
        with self.init_scope():
            self.bn1 = L.BatchNormalization(in_channels)
            self.conv1 = L.Convolution2D(
                in_channels, mid_channels, 1, 1, 0, initialW=initialW,
                nobias=True)
            self.bn2 = L.BatchNormalization(mid_channels)
            self.conv2 = L.Convolution2D(
                mid_channels, mid_channels, 3, stride, 1, initialW=initialW,
                nobias=True)
            self.bn3 = L.BatchNormalization(mid_channels)
            self.conv3 = L.Convolution2D(
                mid_channels, out_channels, 1, 1, 0, initialW=initialW,
                nobias=True)
            self.bn4 = L.BatchNormalization(out_channels)
        self.stride = stride
        self.zero_ch = out_channels - in_channels

    def __call__(self, x):
        h = self.conv1(self.bn1(x))
        h = self.conv2(F.relu(self.bn2(h)))
        h = self.conv3(F.relu(self.bn3(h)))
        h = self.bn4(h)
        batch, h_channel, h_height, h_width = h.shape
        del h_channel
        if self.stride == 2:
            return h + F.concat((
                F.average_pooling_2d(x, 2, 2), chainer.Variable(
                    self.xp.zeros((batch, self.zero_ch, h_height, h_width),
                                  dtype=np.float32))))
        else:
            return h + F.concat((
                x, chainer.Variable(
                    self.xp.zeros((batch, self.zero_ch, h_height, h_width),
                                  dtype=np.float32))))


class PyramidNet(chainer.Chain):

    def __init__(self, n_layers, n_out, alpha=48, stride=(1, 2, 2),
                 layer_names=None):
        super().__init__()
        kwargs = {'initialW': normal.HeNormal(scale=1.0)}
        in_ch = 16

        if (n_layers - 2) % 9 == 0:
            n = (n_layers - 2) // 9
        else:
            raise ValueError(
                'The n_layers argument should be mod({} - 2, 9) == 0,  \
                 but {} was given.'.format(n_layers, n_layers))

        add_ch = alpha / (3 * n)
        temp_ch = in_ch
        with self.init_scope():
            self.conv1 = L.Convolution2D(3, in_ch, 3, 1, 1, nobias=True,
                                         **kwargs)
            self.bn1 = L.BatchNormalization(in_ch)

            for i in range(3):
                name = 'pyramid{}'.format(i + 1)
                pyramid = BuildingPyramidBlock(n, in_ch, temp_ch, add_ch,
                                               stride[i], **kwargs)
                setattr(self, name, pyramid)
                temp_ch += n * add_ch  # temp channel sum
                in_ch = 4 * round(temp_ch)  # next in_ch
            self.bn4 = L.BatchNormalization(in_ch)
            self.fc5 = L.Linear(in_ch, n_out)

        self.functions = collections.OrderedDict([
            ('conv1', [self.conv1]),
            ('pyramid2', [self.pyramid1]),
            ('pyramid3', [self.pyramid2]),
            ('pyramid4', [self.pyramid3]),
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
