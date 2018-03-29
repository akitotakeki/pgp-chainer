import chainer
import chainer.functions as F
import chainer.links as L
import chainer.links.model.vision.resnet as R
from chainer.initializers import normal
import collections
from .resnet_fb import BuildingBlock
from .pgp_lib import pgp, pgp_inv


class ResNet_fb_DConv(chainer.link.Chain):

    def __init__(self, n_layers, n_out, layer_names=None):
        super().__init__()
        kwargs = {'initialW': normal.HeNormal(scale=1.0)}

        if n_layers == 50:
            block = [3, 4, 6, 3]
        elif n_layers == 101:
            block = [3, 4, 23, 3]
        elif n_layers == 152:
            block = [3, 8, 36, 3]
        else:
            raise ValueError('The n_layers argument should be either 50, 101,'
                             ' or 152, but {} was given.'.format(n_layers))

        with self.init_scope():
            self.conv1 = L.Convolution2D(3, 64, 7, 2, 3, nobias=True, **kwargs)
            self.bn1 = L.BatchNormalization(64)
            self.res2 = BuildingBlock(block[0], 64, 64, 256, 1, **kwargs)
            self.res3 = BuildingBlock(block[1], 256, 128, 512, 2, **kwargs)
            self.res4 = BuildingBlock(block[2], 512, 256, 1024, 1, **kwargs)
            self.res5 = BuildingBlock(block[3], 1024, 512, 2048, 1, **kwargs)
            self.fc6 = L.Linear(2048, n_out)

        self.functions = collections.OrderedDict([
            ('conv1', [self.conv1, self.bn1, F.relu]),
            ('pool1', [lambda x: F.max_pooling_2d(x, ksize=3, stride=2)]),
            ('res2', [self.res2]),
            ('res3', [self.res3]),
            ('expand4', [lambda x: pgp(x, 2)]),
            ('res4', [self.res4]),
            ('expand5', [lambda x: pgp(x, 2)]),
            ('res5', [self.res5]),
            ('squeeze5', [lambda x: pgp_inv(x, 4)]),
            ('pool5', [R._global_average_pooling_2d]),
            ('fc6', [self.fc6]),
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

    def extract(self, images, layers=['fc6']):
        self._layer_names = layers
        x = chainer.Variable(self.xp.asarray(images))
        return chainer.cuda.to_cpu(self(x).data)
