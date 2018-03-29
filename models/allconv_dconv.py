#!/usr/bin/env python
# -*- coding: utf-8 -*-

import chainer
import chainer.functions as F
import chainer.links as L
from chainer.initializers import normal
import chainer.links.model.vision.resnet as R
import collections
from chainercv.links import Conv2DBNActiv
from .pgp_lib import pgp, pgp_inv


class AllConv_DConv(chainer.Chain):

    def __init__(self, n_out, layer_names=None):
        super().__init__()
        kwargs = {'initialW': normal.HeNormal(scale=1.0)}

        with self.init_scope():
            self.conv1_1 = Conv2DBNActiv(None, 96, 3, 1, 1, nobias=True, **kwargs)
            self.conv1_2 = Conv2DBNActiv(None, 96, 3, 1, 1, nobias=True, **kwargs)
            self.conv1_3 = Conv2DBNActiv(None, 96, 3, 1, 1, nobias=True, **kwargs)
            self.conv2_1 = Conv2DBNActiv(None, 192, 3, 1, 1, nobias=True, **kwargs)
            self.conv2_2 = Conv2DBNActiv(None, 192, 3, 1, 1, nobias=True, **kwargs)
            self.conv2_3 = Conv2DBNActiv(None, 192, 3, 1, 1, nobias=True, **kwargs)
            self.conv3_1 = Conv2DBNActiv(None, 192, 3, 1, 1, nobias=True, **kwargs)
            self.conv3_2 = Conv2DBNActiv(None, 192, 1, 1, 0, nobias=True, **kwargs)
            self.conv3_3 = Conv2DBNActiv(None, 192, 1, 1, 0, nobias=True, **kwargs)
            self.fc = L.Linear(None, n_out)

        self.functions = collections.OrderedDict([
            ('conv1', [lambda x: F.dropout(x, 0.2),
                       self.conv1_1, self.conv1_2, lambda x: pgp(x, 2),
                       self.conv1_3, F.dropout]),
            ('conv2', [self.conv2_1, self.conv2_2, lambda x: pgp(x, 2),
                       self.conv2_3, F.dropout]),
            ('conv3', [self.conv3_1, self.conv3_2, self.conv3_3]),
            ('squeeze3', [lambda x: pgp_inv(x, 4)]),
            ('pool3', [R._global_average_pooling_2d]),
            ('fc', [self.fc]),
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

    def extract(self, images, layers=['fc']):
        self._layer_names = layers
        x = chainer.Variable(self.xp.asarray(images))
        return chainer.cuda.to_cpu(self(x).data)
