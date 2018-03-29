import collections
import itertools as it
import numpy as np
from numpy.random import randint, seed

import chainer
import chainer.functions as F
from chainer.initializers import normal
import chainer.links as L
import chainer.links.model.vision.resnet as R
from .preresnet import PreBuildingBlock


def _gen_kernel(size, grid):
    stride = int(size / grid)
    x, y = np.arange(stride) * 2, np.arange(stride) * 2
    xx, yy = np.meshgrid(x, y)
    ones = np.c_[xx.ravel(), yy.ravel()] + randint(0, grid,
                                                   (stride * stride, 2))
    zeros = np.zeros((size, size))
    for (i, j) in ones:
        zeros[i, j] = 1
    return zeros


def _gen_lattice(size, grid, i, j):
    d = np.zeros((size, size))
    d[i::grid, j::grid] = 1
    return d


def _squeeze(x, r=2):
    in_batch, in_channel, in_height, in_width = x.data.shape
    out_batch, out_channel, out_height, out_width = int(in_batch / (r ** 2)), in_channel, r * in_height, r * in_width
    h = x.reshape(r, r, out_batch, in_channel, in_height, in_width)
    h = h.transpose(2, 3, 4, 0, 5, 1)
    return h.reshape(out_batch, out_channel, out_height, out_width)


def _expand(x, r=2):
    in_batch, in_channel, in_height, in_width = x.data.shape
    out_batch, out_channel, out_height, out_width = int(in_batch * (r ** 2)), in_channel, int(in_height / r),  int(in_width / r)
    h = x.reshape(in_batch, in_channel, out_height, r, out_width, r)
    h = h.transpose(3, 5, 0, 1, 2, 4)
    return h.reshape(out_batch, out_channel, out_height, out_width)


def batch_squeeze(x, r, n_kernel):
    assert n_kernel % 4 == 0

    in_batch, in_channel, in_height, in_width = x.data.shape
    h = x.reshape(n_kernel // 4,  4 * in_batch // n_kernel,
                  in_channel, in_height, in_width)
    h = F.stack(list(map(lambda x: _squeeze(x, r=r), h)))
    return h.reshape(in_batch // (r ** 2), in_channel,
                     in_height * r, in_width * r)


def batch_expansion(x, r, n_kernel):
    assert n_kernel % 4 == 0

    in_batch, in_channel, in_height, in_width = x.data.shape
    h = x.reshape(n_kernel // 4, 4 * in_batch // n_kernel,
                  in_channel, in_height, in_width)
    h = F.stack(list(map(lambda x: _expand(x, r=r), h)))
    return h.reshape(in_batch * r ** 2, in_channel,
                     in_height // r, in_width // r)


class Expand(chainer.link.Chain):

    def __init__(self, n_kernel, size, grid=2, seed_value=0, random=False,
                 lattice=False, width=tuple()):
        super().__init__()
        if lattice:
            self.kernel = [_gen_lattice(size, grid, i, j) for i in range(grid)
                           for j in range(grid)]
        else:
            seed(seed_value)
            self.kernel = [_gen_kernel(size, grid) for i in range(n_kernel)]
        self.grid = grid
        self.n_kernel = n_kernel
        self.random = random
        if random:
            self.size = size
        self.width = width

    def __call__(self, x):
        if self.random:
            self.kernel = [_gen_kernel(self.size, self.grid)
                           for i in range(self.n_kernel)]

        h_list = [x]
        in_batch, in_ch, in_height, in_width = x.shape
        for w in self.width:
            h_list.append(x.reshape(in_batch, in_ch, in_height // (2 * w), 2, w, in_width // (2 * w), 2, w).transpose(0, 1, 2, 4, 3, 5, 7, 6).reshape(in_batch, in_ch, in_height, in_width))

        inf = 1 - np.stack(self.kernel)
        inf[inf == 1] = -np.inf

        h = list()
        for _h in h_list:
            h.append(tuple(_h + chainer.cuda.to_gpu(
                     inf[i]) for i in range(len(self.kernel))))
        h = F.concat(it.chain(*h), axis=0)

        '''
        if self.width:
            in_batch, in_ch, in_height, in_width = x.shape
            h0 = x.reshape(in_batch, in_ch, in_height // 4, 2, 2, in_width // 4, 2, 2).transpose(0, 1, 2, 4, 3, 5, 7, 6).reshape(in_batch, in_ch, in_height, in_width)
        inf = 1 - np.stack(self.kernel)
        inf[inf == 1] = -np.inf
        if self.width and self.n_kernel == 8:
            h0 = tuple(h0 + chainer.cuda.to_gpu(
                       inf[i]) for i in range(len(self.kernel)))
            h = tuple(h + chainer.cuda.to_gpu(
                      inf[i]) for i in range(len(self.kernel)))
            h0 = F.concat(h0, axis=0)
            h = F.concat(h, axis=0)
            h = F.concat((h0, h), axis=0)
        elif self.width:
            h = tuple(h0 + chainer.cuda.to_gpu(
                      inf[i]) for i in range(len(self.kernel)))
            h = F.concat(h, axis=0)
        else:
            h = tuple(h + chainer.cuda.to_gpu(
                      inf[i]) for i in range(len(self.kernel)))
            h = F.concat(h, axis=0)
        '''
        return F.max_pooling_2d(h, ksize=self.grid)


class PreBuildingBlock_shareBN(chainer.Chain):

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
                 out_channels, stride, r=2, n_kernel=4, initialW=None):
        super().__init__()
        with self.init_scope():
            self.a = PreBottleneckA_shareBN(
                in_channels, mid_channels, out_channels, stride, r, n_kernel,
                initialW)
            self.forward = [self.a]
            for i in range(n_layer - 1):
                name = 'b{}'.format(i + 1)
                bottleneck = PreBottleneckB_shareBN(
                    out_channels, mid_channels, r, n_kernel,
                    initialW)
                setattr(self, name, bottleneck)
                self.forward.append(bottleneck)

    def __call__(self, x):
        for l in self.forward:
            x = l(x)
        return x


class PreBottleneckA_shareBN(chainer.Chain):

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
                 stride=2, r=2, n_kernel=4, initialW=None):
        super().__init__()
        with self.init_scope():
            self.bn1 = L.BatchNormalization(in_channels)
            self.conv1 = L.Convolution2D(
                in_channels, mid_channels, 1, stride, 0, initialW=initialW,
                nobias=True)
            self.bn2 = L.BatchNormalization(mid_channels)
            self.conv2 = L.Convolution2D(
                mid_channels, mid_channels, 3, 1, 1, initialW=initialW,
                nobias=True)
            self.bn3 = L.BatchNormalization(mid_channels)
            self.conv3 = L.Convolution2D(
                mid_channels, out_channels, 1, 1, 0, initialW=initialW,
                nobias=True)
            self.conv4 = L.Convolution2D(
                in_channels, out_channels, 1, stride, 0, initialW=initialW,
                nobias=True)
        self.r = r
        self.n_kernel = n_kernel

    def __call__(self, x):
        h0 = batch_squeeze(x, self.r, self.n_kernel)
        h0 = F.relu(self.bn1(h0))
        h0 = batch_expansion(h0, self.r, self.n_kernel)

        h1 = self.conv1(h0)

        h1 = batch_squeeze(h1, self.r, self.n_kernel)
        h1 = F.relu(self.bn2(h1))
        h1 = batch_expansion(h1, self.r, self.n_kernel)
        h1 = self.conv2(h1)

        h1 = batch_squeeze(h1, self.r, self.n_kernel)
        h1 = F.relu(self.bn3(h1))
        h1 = batch_expansion(h1, self.r, self.n_kernel)
        h1 = self.conv3(h1)

        h2 = self.conv4(h0)
        return h1 + h2


class PreBottleneckB_shareBN(chainer.Chain):

    """A bottleneck layer that maintains the resolution of the feature map.

    Args:
        in_channels (int): Number of channels of input and output arrays.
        mid_channels (int): Number of channels of intermediate arrays.
        initialW (4-D array): Initial weight value used in
            the convolutional layers.
    """

    def __init__(self, in_channels, mid_channels, r=2, n_kernel=4,
                 initialW=None):
        super().__init__()
        with self.init_scope():
            self.bn1 = L.BatchNormalization(in_channels)
            self.conv1 = L.Convolution2D(
                in_channels, mid_channels, 1, 1, 0, initialW=initialW,
                nobias=True)
            self.bn2 = L.BatchNormalization(mid_channels)
            self.conv2 = L.Convolution2D(
                mid_channels, mid_channels, 3, 1, 1, initialW=initialW,
                nobias=True)
            self.bn3 = L.BatchNormalization(mid_channels)
            self.conv3 = L.Convolution2D(
                mid_channels, in_channels, 1, 1, 0, initialW=initialW,
                nobias=True)
        self.r = r
        self.n_kernel = n_kernel

    def __call__(self, x):
        h = batch_squeeze(x, self.r, self.n_kernel)
        h = F.relu(self.bn1(h))
        h = batch_expansion(h, self.r, self.n_kernel)
        h = self.conv1(h)

        h = batch_squeeze(h, self.r, self.n_kernel)
        h = F.relu(self.bn2(h))
        h = batch_expansion(h, self.r, self.n_kernel)
        h = self.conv2(h)

        h = batch_squeeze(h, self.r, self.n_kernel)
        h = F.relu(self.bn3(h))
        h = batch_expansion(h, self.r, self.n_kernel)
        h = self.conv3(h)
        return h + x


class PreResNet_s3(chainer.Chain):

    def __init__(self, n_layers, n_out, n_kernel=4, grid=2, seed_value=0,
                 random=False, lattice=False, width=tuple(), layer_names=None):
        super().__init__()
        kwargs = {'initialW': normal.HeNormal(scale=1.0)}

        if n_layers == 47:
            block = [5, 5, 5]
        elif n_layers == 74:
            block = [8, 8, 8]
        elif n_layers == 110:
            block = [12, 12, 12]
        elif n_layers == 164:
            block = [18, 18, 18]
        else:
            raise ValueError(
                'The n_layers argument should be mod({} - 2, 3) == 0,  \
                 but {} was given.'.format(n_layers, n_layers))

        with self.init_scope():
            self.conv1 = L.Convolution2D(3, 16, 3, 1, 1, **kwargs)
            self.res2 = PreBuildingBlock(block[0], 16, 16, 64, 1, **kwargs)
            self.expand3 = Expand(n_kernel, 32, grid=grid,
                                  seed_value=seed_value, random=random,
                                  lattice=lattice, width=width)
            self.res3 = PreBuildingBlock_shareBN(block[1], 64, 32, 128, 1, 2,
                                                 n_kernel, **kwargs)
            self.expand4 = Expand(n_kernel, 16, grid=grid,
                                  seed_value=seed_value, random=random,
                                  lattice=lattice, width=width)
            self.res4 = PreBuildingBlock_shareBN(block[2], 128, 64, 256, 1, 4,
                                                 n_kernel, **kwargs)
            self.bn4 = L.BatchNormalization(256)
            self.fc5 = L.Linear(256, n_out)

        self.functions = collections.OrderedDict([
            ('conv1', [self.conv1]),
            ('res2', [self.res2]),
            ('expand3', [self.expand3]),
            ('res3', [self.res3]),
            ('expand4', [self.expand4]),
            ('res4', [self.res4]),
            ('squeeze4', [lambda x: batch_squeeze(x, 4, n_kernel)]),
            ('pool4', [self.bn4, F.relu,
                       lambda x: F.split_axis(x, (n_kernel // 4) ** 2, axis=0),
                       lambda x: F.concat(x, axis=2),
                       R._global_average_pooling_2d]),
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
