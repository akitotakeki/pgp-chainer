import collections

import chainer
import chainer.functions as F
from chainer.initializers import normal
import chainer.links as L
from .preresnet import PreBuildingBlock
from .pgp_lib import pgp


class PreResNet_PGP(chainer.Chain):

    def __init__(self, n_layers, n_out, layer_names=None):
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
            self.res2 = PreBuildingBlock(block[0], 16, 16, 64, 1, **kwargs)
            self.res3 = PreBuildingBlock(block[1], 64, 32, 128, 1, **kwargs)
            self.res4 = PreBuildingBlock(block[2], 128, 64, 256, 1, **kwargs)
            self.bn4 = L.BatchNormalization(256)
            self.fc5 = L.Linear(256, n_out)

        self.functions = collections.OrderedDict([
            ('conv1', [self.conv1]),
            ('res2', [self.res2]),
            ('expand3', [lambda x: pgp(x, 2)]),
            ('res3', [self.res3]),
            ('expand4', [lambda x: pgp(x, 2)]),
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
        h = F.stack(F.split_axis(h, 16, axis=0))
        h = F.average(F.softmax(h, axis=2), axis=0)
        return chainer.cuda.to_cpu(h.data)
