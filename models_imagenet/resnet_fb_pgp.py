import chainer
import chainer.functions as F
import chainer.links as L
from chainer.initializers import normal
import collections
from .resnet_fb import BuildingBlock
from .pgp_lib import pgp


class BuildingBlock_alter(chainer.link.Chain):

    """A building block that consists of several Bottleneck layers.

    Args:
        n_layer (int): Number of layers used in the building block.
        in_channels (int): Number of channels of input arrays.
        mid_channels (int): Number of channels of intermediate arrays.
        out_channels (int): Number of channels of output arrays.
        stride (int or tuple of ints): Stride of filter application.
        initialW (4-D array): Initial weight value used in
            the convolutional layers.
    """

    def __init__(self, n_layer, in_channels, mid_channels,
                 out_channels, stride, initialW=None):
        super(BuildingBlock_alter, self).__init__()
        with self.init_scope():
            self.a = BottleneckA(
                in_channels, mid_channels, out_channels, stride, initialW)
            self._forward = ["a"]
            for i in range(n_layer - 1):
                name = 'b{}'.format(i + 1)
                bottleneck = BottleneckB(out_channels, mid_channels, initialW)
                setattr(self, name, bottleneck)
                self._forward.append(name)

    def __call__(self, x):
        for name in self._forward:
            layer = getattr(self, name)
            x = layer(x)
        return x

    @property
    def forward(self):
        return [getattr(self, name) for name in self._forward]


class BottleneckA(chainer.link.Chain):

    """A bottleneck layer that reduces the resolution of the feature map.

    Args:
        in_channels (int): Number of channels of input arrays.
        mid_channels (int): Number of channels of intermediate arrays.
        out_channels (int): Number of channels of output arrays.
        stride (int or tuple of ints): Stride of filter application.
        initialW (4-D array): Initial weight value used in
            the convolutional layers.
    """

    def __init__(self, in_channels, mid_channels, out_channels,
                 stride=2, initialW=None):
        super(BottleneckA, self).__init__()

        self.stride = stride
        with self.init_scope():
            self.conv1 = L.Convolution2D(
                in_channels, mid_channels, 1, 1, 0, initialW=initialW,
                nobias=True)
            self.bn1 = L.BatchNormalization(mid_channels)
            self.conv2 = L.Convolution2D(
                mid_channels, mid_channels, 3, 1, 1,
                initialW=initialW, nobias=True)
            self.bn2 = L.BatchNormalization(mid_channels)
            self.conv3 = L.Convolution2D(
                mid_channels, out_channels, 1, 1, 0, initialW=initialW,
                nobias=True)
            self.bn3 = L.BatchNormalization(out_channels)
            self.conv4 = L.Convolution2D(
                in_channels, out_channels, 1, 1, 0, initialW=initialW,
                nobias=True)
            self.bn4 = L.BatchNormalization(out_channels)

    def __call__(self, x):
        h1 = F.relu(self.bn1(self.conv1(x)))
        h1 = F.relu(self.bn2(self.conv2(h1)))
        if self.stride == 2:
            h1 = self.bn3(self.conv3(pgp(h1, 2)))
            h2 = self.bn4(self.conv4(pgp(x, 2)))
        else:
            h1 = self.bn3(self.conv3(h1))
            h2 = self.bn4(self.conv4(x))
        return F.relu(h1 + h2)


class BottleneckB(chainer.link.Chain):

    """A bottleneck layer that maintains the resolution of the feature map.

    Args:
        in_channels (int): Number of channels of input and output arrays.
        mid_channels (int): Number of channels of intermediate arrays.
        initialW (4-D array): Initial weight value used in
            the convolutional layers.
    """

    def __init__(self, in_channels, mid_channels, initialW=None):
        super(BottleneckB, self).__init__()
        with self.init_scope():
            self.conv1 = L.Convolution2D(
                in_channels, mid_channels, 1, 1, 0, initialW=initialW,
                nobias=True)
            self.bn1 = L.BatchNormalization(mid_channels)
            self.conv2 = L.Convolution2D(
                mid_channels, mid_channels, 3, 1, 1, initialW=initialW,
                nobias=True)
            self.bn2 = L.BatchNormalization(mid_channels)
            self.conv3 = L.Convolution2D(
                mid_channels, in_channels, 1, 1, 0, initialW=initialW,
                nobias=True)
            self.bn3 = L.BatchNormalization(in_channels)

    def __call__(self, x):
        h = F.relu(self.bn1(self.conv1(x)))
        h = F.relu(self.bn2(self.conv2(h)))
        h = self.bn3(self.conv3(h))
        return F.relu(h + x)


class ResNet_fb_PGP(chainer.link.Chain):

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
            self.res4 = BuildingBlock_alter(block[2], 512, 256, 1024, 2, **kwargs)
            self.res5 = BuildingBlock_alter(block[3], 1024, 512, 2048, 2, **kwargs)
            self.fc6 = L.Linear(2048, n_out)

        self.functions = collections.OrderedDict([
            ('conv1', [self.conv1, self.bn1, F.relu]),
            ('pool1', [lambda x: F.max_pooling_2d(x, ksize=3, stride=2)]),
            ('res2', [self.res2]),
            ('res3', [self.res3]),
            ('res4', [self.res4]),
            ('res5', [self.res5]),
            ('pool5', [lambda x: F.average(x, axis=(2, 3))]),
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
        h = self(x).data
        _len, _cls = h.shape
        h = F.average(F.reshape(h, (16, _len // 16, _cls)), axis=0)
        return chainer.cuda.to_cpu(h.data)
