import collections

import chainer
import chainer.functions as F
import chainer.links as L
from .densenet import DenseBlock
from .pgp_lib import pgp, pgp_inv


class TransitionLayer(chainer.Chain):

    def __init__(self, in_ch, out_ch, pool=True):
        super(TransitionLayer, self).__init__()
        self.pool = pool
        with self.init_scope():
            self.bn = L.BatchNormalization(in_ch)
            self.conv = L.Convolution2D(in_ch, out_ch, 1, 1, 0)

    def __call__(self, x):
        x = self.conv(F.relu(self.bn(x)))
        if self.pool:
            x = F.average_pooling_2d(x, 2, stride=2)
        return x


class DenseNetBC_DConv(chainer.Chain):

    def __init__(self, n_out, stage_sizes, growth_rate, reduction=0.5,
                 pool=False, layer_names=None):
        super().__init__()

        ch = growth_rate * 2
        self.n_blocks = len(stage_sizes)

        with self.init_scope():
            self.conv1 = L.Convolution2D(
                3, ch, 3, 1, 1, initialW=chainer.initializers.HeNormal(),
                nobias=True)

            for i, s in enumerate(stage_sizes):
                block = DenseBlock(s, ch, growth_rate)

                ch += s * growth_rate
                setattr(self, 'block{}'.format(i), block)

                if i + 1 < len(stage_sizes):
                    ch2 = int(ch * reduction)
                    trans = TransitionLayer(ch, ch2, pool=pool)
                    ch = ch2
                    setattr(self, 'trans{}'.format(i), trans)

            self.fc_bn = L.BatchNormalization(ch)
            self.fc = L.Linear(ch, n_out)

        self.functions = collections.OrderedDict([
            ('conv1', [self.conv1]),
            ('block2', [self.block0]),
            ('trans2', [self.trans0]),
            ('expand2', [lambda x: pgp(x, 2)]),
            ('block3', [self.block1]),
            ('trans3', [self.trans1]),
            ('expand3', [lambda x: pgp(x, 2)]),
            ('block4', [self.block2]),
            ('squeeze4', [lambda x: pgp_inv(x, 4)]),
            ('pool4', [self.fc_bn, F.relu, lambda x: F.average(x, axis=(2, 3))]),
            ('fc5', [self.fc]),
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
