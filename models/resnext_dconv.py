#!/usr/bin/env python
# -*- coding: utf-8 -*-

import chainer
import chainer.functions as F
import chainer.links as L
from chainer.initializers import normal
import collections
import chainer.links.model.vision.resnet as R
from .resnext import BuildingBlock
from .pgp_lib import pgp, pgp_inv


class ResNeXt_DConv(chainer.Chain):

    def __init__(self, n_layers, n_out, cardinality=8, base_width=64,
                 widen_factor=4, layer_names=None):
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
            self.res3 = BuildingBlock(block[1], 64 * widen_factor, 128 * widen_factor, cardinality, base_width, widen_factor, 1, **kwargs)
            self.res4 = BuildingBlock(block[2], 128 * widen_factor, 256 * widen_factor, cardinality, base_width, widen_factor, 1, **kwargs)
            self.fc5 = L.Linear(256 * widen_factor, n_out)

        self.functions = collections.OrderedDict([
            ('conv1', [self.conv1, self.bn1, F.relu]),
            ('res2', [self.res2]),
            ('expand3', [lambda x: pgp(x, 2)]),
            ('res3', [self.res3]),
            ('expand4', [lambda x: pgp(x, 2)]),
            ('res4', [self.res4]),
            ('squeeze4', [lambda x: pgp_inv(x, 4)]),
            ('pool4', [R._global_average_pooling_2d]),
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
