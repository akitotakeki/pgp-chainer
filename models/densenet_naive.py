import collections

import math
import chainer
import chainer.functions as F
import chainer.links as L
from chainer.initializers import normal


class DenseCluster(chainer.Chain):

    def __init__(self, in_channels, out_channels, bottleneck=False,
                 initialW=None):
        super().__init__()
        with self.init_scope():
            if bottleneck:
                self.bn1 = L.BatchNormalization(in_channels)
                self.conv1 = L.Convolution2D(
                    in_channels, out_channels * 4, 1, 1, 0, initialW=initialW,
                    nobias=True)
                self.bn2 = L.BatchNormalization(out_channels * 4)
                self.conv2 = L.Convolution2D(
                    out_channels * 4, out_channels, 3, 1, 1, initialW=initialW,
                    nobias=True)
            else:
                self.bn1 = L.BatchNormalization(in_channels)
                self.conv1 = L.Convolution2D(
                    in_channels, out_channels, 3, 1, 1, initialW=initialW,
                    nobias=True)
        self.bottleneck = bottleneck

    def __call__(self, x):
        if self.bottleneck:
            h = self.conv1(F.relu(self.bn1(x)))
            h = self.conv2(F.relu(self.bn2(h)))
        else:
            h = self.conv1(F.relu(self.bn1(x)))
        return F.concat((x, h))


class BuildingDenseBlock(chainer.link.Chain):

    def __init__(self, in_channels, growth_rate, n_layer, bottleneck=False,
                 initialW=None):
        super().__init__()
        with self.init_scope():
            self.forward = list()
            for i in range(n_layer):
                name = 'b{}'.format(i + 1)
                cluster = DenseCluster(in_channels + i * growth_rate,
                                       growth_rate, bottleneck=bottleneck,
                                       initialW=initialW)
                setattr(self, name, cluster)
                self.forward.append(cluster)

    def __call__(self, x):
        for l in self.forward:
            x = l(x)
        return x


class Transition(chainer.Chain):

    def __init__(self, in_channels, reduction=1.0, stride=2,
                 initialW=None):
        super().__init__()
        with self.init_scope():
            self.bn = L.BatchNormalization(in_channels)
            self.conv = L.Convolution2D(
                in_channels, math.floor(in_channels * reduction), 1,
                initialW=initialW, nobias=True)
        self.stride = stride

    def __call__(self, x):
        h = F.relu(self.bn(x))
        h = self.conv(h)
        h = F.average_pooling_2d(h, self.stride)
        return h


class DenseNet(chainer.Chain):

    def __init__(self, n_layer=12, growth_rate=12,
                 n_class=10, in_ch=16, block=3,
                 bottleneck=False, reduction=1.0, stride=[2, 2],
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
            ('block3', [self.dense2]),
            ('trans3', [self.trans2]),
            ('block4', [self.dense3]),
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
