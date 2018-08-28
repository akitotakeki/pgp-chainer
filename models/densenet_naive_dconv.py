import collections

import math
import chainer
import chainer.functions as F
import chainer.links as L
from chainer.initializers import normal
from .densenet_naive import BuildingDenseBlock, Transition
from .pgp_lib import pgp, pgp_inv


class DenseNet_DConv(chainer.Chain):

    def __init__(self, n_layer=12, growth_rate=12,
                 n_class=10, in_ch=16, block=3,
                 bottleneck=False, reduction=1.0, stride=[1, 1],
                 layer_names=None):
        """DenseNet definition.
        Args:
            n_layer: Number of convolution layers in one dense block.
                If n_layer=12, the network is made out of 40 (12*3+4) layers.
                If n_layer=32, the network is made out of 100 (32*3+4) layers.
            growth_rate: Number of output feature maps of each convolution
                layer in dense blocks, which is difined as k in the paper.
            n_class: Output class.
            dropout_ratio: Dropout ratio.
            in_ch: Number of output feature maps of first convolution layer.
            block: Number of dense block.
        """
        super().__init__()

        kwargs = {'initialW': normal.HeNormal(scale=1.0)}

        with self.init_scope():
            self.conv = L.Convolution2D(3, in_ch, 3, 1, 1, nobias=True,
                                        **kwargs)
            # self.forward = list()
            for i in range(block):
                name = 'dense{}'.format(i + 1)
                dense = BuildingDenseBlock(in_ch, growth_rate, n_layer,
                                           bottleneck=bottleneck, **kwargs)
                setattr(self, name, dense)
                # self.forward.append(dense)
                in_ch = in_ch + n_layer * growth_rate
                if not i == block - 1:
                    name = 'trans{}'.format(i + 1)
                    trans = Transition(in_ch, reduction=reduction,
                                       stride=stride[i], **kwargs)
                    setattr(self, name, trans)
                    # self.forward.append(trans)
                    in_ch = math.floor(in_ch * reduction)
            self.bn = L.BatchNormalization(in_ch)
            self.fc = L.Linear(in_ch, n_class)

        self.functions = collections.OrderedDict([
            ('conv1', [self.conv]),
            ('block2', [self.dense1]),
            ('trans2', [self.trans1]),
            ('expand2', [lambda x: pgp(x, 2)]),
            ('block3', [self.dense2]),
            ('trans3', [self.trans2]),
            ('expand3', [lambda x: pgp(x, 2)]),
            ('block4', [self.dense3]),
            ('expand4', [lambda x: pgp_inv(x, 4)]),
            ('pool4', [self.bn, F.relu, lambda x: F.average(x, axis=(2, 3))]),
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
