#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import chainer
from chainer import cuda
from chainer import configuration
import chainer.functions as F
import chainer.links as L
from chainer.initializers import normal
import collections


class Mul(chainer.function.Function):

    def forward(self, inputs):
        x1, x2 = inputs[:2]
        xp = cuda.get_array_module(x1)
        alpha = xp.ones(x1.shape, dtype=x1.dtype) * 0.5
        if configuration.config.train:
            for i in range(len(alpha)):
                alpha[i] = xp.random.rand()
        return x1 * alpha + x2 * (xp.ones(x1.shape, dtype=x1.dtype) - alpha),

    def backward(self, inputs, grad_outputs):
        gx = grad_outputs[0]
        xp = cuda.get_array_module(gx)
        beta = xp.empty(gx.shape, dtype=gx.dtype)
        for i in range(len(beta)):
            beta[i] = xp.random.rand()
        return gx * beta, gx * (xp.ones(gx.shape, dtype=gx.dtype) - beta)


def mul(x1, x2):
    return Mul()(x1, x2)


class BuildingShakeBlock(chainer.link.Chain):

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
        super(BuildingShakeBlock, self).__init__()
        with self.init_scope():
            self.a = ShakeA(in_channels, out_channels, stride, initialW)
            self._forward = ["a"]
            for i in range(n_layer - 1):
                name = 'b{}'.format(i + 1)
                basic = ShakeB(out_channels, initialW)
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


class ShakeA(chainer.link.Chain):

    def __init__(self, in_channels, out_channels, stride=2, initialW=None):
        super(ShakeA, self).__init__()

        with self.init_scope():
            self.branch1 = RCBRCB(in_channels, out_channels, stride=stride,
                                  initialW=initialW)
            self.branch2 = RCBRCB(in_channels, out_channels, stride=stride,
                                  initialW=initialW)
            self.conv1 = L.Convolution2D(
                in_channels, out_channels // 2, 1, stride, 0,
                initialW=initialW, nobias=True)
            self.conv2 = L.Convolution2D(
                in_channels, out_channels // 2, 1, stride, 0,
                initialW=initialW, nobias=True)
            self.bn = L.BatchNormalization(out_channels)

    def __call__(self, x):
        h1 = self.branch1(x)
        h2 = self.branch2(x)

        x0 = F.relu(x)
        x1 = self.conv1(x0)
        x2 = self.zero_pads(self.zero_pads(x0, 1, 2), 1, 3)[:, :, 1:, 1:]
        x2 = self.conv2(x2)
        h0 = F.concat((x1, x2))
        h0 = self.bn(h0)
        return mul(h1, h2) + h0

    def zero_pads(self, x, pad, where):
        sizes = list(x.data.shape)
        sizes[where] = pad
        pad_mat = chainer.Variable(chainer.cuda.to_gpu(self.xp.zeros(sizes, dtype=np.float32), device=chainer.cuda.get_device_from_array(x.data)))
        return F.concat((pad_mat, x), axis=where)


class ShakeB(chainer.link.Chain):

    def __init__(self, in_channels, initialW=None):
        super(ShakeB, self).__init__()
        with self.init_scope():
            self.branch1 = RCBRCB(in_channels, in_channels, initialW=initialW)
            self.branch2 = RCBRCB(in_channels, in_channels, initialW=initialW)

    def __call__(self, x):
        h1 = self.branch1(x)
        h2 = self.branch2(x)
        return mul(h1, h2) + x


class RCBRCB(chainer.link.Chain):

    def __init__(self, in_channels, out_channels, stride=1, initialW=None):
        super(RCBRCB, self).__init__()
        with self.init_scope():
            # self.bn1 = L.BatchNormalization(in_channels)
            self.conv1 = L.Convolution2D(
                in_channels, out_channels, 3, stride, 1, initialW=initialW,
                nobias=True)
            self.bn1 = L.BatchNormalization(out_channels)
            self.conv2 = L.Convolution2D(
                out_channels, out_channels, 3, 1, 1, initialW=initialW,
                nobias=True)
            self.bn2 = L.BatchNormalization(out_channels)

    def __call__(self, x):
        h = self.conv1(F.relu(x))
        h = self.conv2(F.relu(self.bn1(h)))
        h = self.bn2(h)
        return h


class ShakeShake(chainer.Chain):

    def __init__(self, n_layers, n_out, k=32, layer_names=None):
        super().__init__()
        kwargs = {'initialW': normal.HeNormal(scale=1.0)}

        if (n_layers - 2) % 6 == 0:
            block = [(n_layers - 2) // 6] * 3
        else:
            raise ValueError(
                'The n_layers argument should be mod({} - 4, 6) == 0,  \
                 but {} was given.'.format(n_layers, n_layers))

        with self.init_scope():
            self.conv1 = L.Convolution2D(3, 16, 3, 1, 1, nobias=True, **kwargs)
            self.bn1 = L.BatchNormalization(16)
            self.res2 = BuildingShakeBlock(
                block[0], 16, k, 1, **kwargs)
            self.res3 = BuildingShakeBlock(
                block[1], k, 2 * k, 2, **kwargs)
            self.res4 = BuildingShakeBlock(
                block[2], 2 * k, 4 * k, 2, **kwargs)
            self.fc5 = L.Linear(4 * k, n_out)

        self.functions = collections.OrderedDict([
            ('conv1', [self.conv1, self.bn1]),
            ('res2', [self.res2]),
            ('res3', [self.res3]),
            ('res4', [self.res4]),
            ('pool4', [F.relu, lambda x: F.average(x, axis=(2, 3))]),
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
