#!/usr/bin/env python
# -*- coding: utf-8 -*-

import chainer
import chainer.functions as F
import chainer.links as L
from chainer.initializers import normal
import collections
from .pyramidnet import BuildingPyramidBlock
from .pgp_lib import pgp, pgp_inv


class PyramidNet_DConv(chainer.Chain):

    def __init__(self, n_layers, n_out, alpha=48, stride=(1, 1, 1),
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
            ('expand3', [lambda x: pgp(x, 2)]),
            ('pyramid3', [self.pyramid2]),
            ('expand4', [lambda x: pgp(x, 2)]),
            ('pyramid4', [self.pyramid3]),
            ('squeeze4', [lambda x: pgp_inv(x, 4)]),
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
