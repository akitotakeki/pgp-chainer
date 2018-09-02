#!/usr/bin/env python
# -*- coding: utf-8 -*-

import chainer
import chainer.functions as F
import chainer.links as L
from chainer.initializers import normal
import collections
from .pgp_lib import pgp


class PreBuildingBasicBlock(chainer.link.Chain):

    """A building block that consists of several Basic layers.

    Args:
        n_layer (int): Number of layers used in the building block.
        in_channels (int): Number of channels of input arrays.
        out_channels (int): Number of channels of output arrays.
        stride (int or tuple of ints): Stride of filter application.
        initialW (4-D array): Initial weight value used in
            the convolutional layers.
    """

    def __init__(self, n_layer, in_channels,
                 out_channels, stride, initialW=None):
        super(PreBuildingBasicBlock, self).__init__()
        with self.init_scope():
            self.a = PreBasicA(in_channels, out_channels, stride, initialW)
            self._forward = ["a"]
            for i in range(n_layer - 1):
                name = 'b{}'.format(i + 1)
                basic = PreBasicB(out_channels, initialW)
                setattr(self, name, basic)
                self._forward.append(name)

    def __call__(self, x):
        for name in self._forward:
            layer = getattr(self, name)
            x = layer(x)
        return x

    @property
    def forward(self):
        return [getattr(self, name) for name in self._forward]


class PreBasicA(chainer.link.Chain):

    def __init__(self, in_channels, out_channels, stride=2, initialW=None):
        super(PreBasicA, self).__init__()

        self.stride = stride
        with self.init_scope():
            self.bn1 = L.BatchNormalization(in_channels)
            self.conv1 = L.Convolution2D(
                in_channels, out_channels, 3, 1, 1, initialW=initialW,
                nobias=True)
            self.bn2 = L.BatchNormalization(out_channels)
            self.conv2 = L.Convolution2D(
                out_channels, out_channels, 3, 1, 1, initialW=initialW,
                nobias=True)
            self.conv3 = L.Convolution2D(
                in_channels, out_channels, 1, 1, 0, initialW=initialW,
                nobias=True)

    def __call__(self, x):
        h0 = F.relu(self.bn1(x))
        h1 = self.conv1(h0)
        if self.stride == 2:
            h1 = self.conv2(F.dropout(pgp(F.relu(self.bn2(h1)), 2), 0.3))
            h2 = self.conv3(pgp(h0, 2))
        else:
            h1 = self.conv2(F.dropout(F.relu(self.bn2(h1)), 0.3))
            h2 = self.conv3(h0)
        return h1 + h2


class PreBasicB(chainer.link.Chain):

    def __init__(self, in_channels, initialW=None):
        super(PreBasicB, self).__init__()
        with self.init_scope():
            self.bn1 = L.BatchNormalization(in_channels)
            self.conv1 = L.Convolution2D(
                in_channels, in_channels, 3, 1, 1, initialW=initialW,
                nobias=True)
            self.bn2 = L.BatchNormalization(in_channels)
            self.conv2 = L.Convolution2D(
                in_channels, in_channels, 3, 1, 1, initialW=initialW,
                nobias=True)

    def __call__(self, x):
        h = self.conv1(F.relu(self.bn1(x)))
        h = self.conv2(F.dropout(F.relu(self.bn2(h)), 0.3))
        return h + x


class WideResNet_PGP(chainer.Chain):

    def __init__(self, n_layers, n_out, k=1, layer_names=None):
        super().__init__()
        kwargs = {'initialW': normal.HeNormal(scale=1.0)}

        if (n_layers - 4) % 6 == 0:
            block = [(n_layers - 4) // 6] * 3
        else:
            raise ValueError(
                'The n_layers argument should be mod({} - 4, 6) == 0,  \
                 but {} was given.'.format(n_layers, n_layers))

        with self.init_scope():
            self.conv1 = L.Convolution2D(3, 16, 3, 1, 1, **kwargs)
            self.res2 = PreBuildingBasicBlock(
                block[0], 16, 16 * k, 1, **kwargs)
            self.res3 = PreBuildingBasicBlock(
                block[1], 16 * k, 32 * k, 2, **kwargs)
            self.res4 = PreBuildingBasicBlock(
                block[2], 32 * k, 64 * k, 2, **kwargs)
            self.bn4 = L.BatchNormalization(64 * k)
            self.fc5 = L.Linear(64 * k, n_out)

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
        h = self(x).data
        _len, _cls = h.shape
        h = F.average(F.reshape(h, (16, _len // 16, _cls)), axis=0)
        return chainer.cuda.to_cpu(h.data)
