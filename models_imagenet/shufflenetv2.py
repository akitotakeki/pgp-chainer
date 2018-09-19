import chainer
import chainer.functions as F
import chainer.links as L
from chainer.initializers import normal
import collections
from .shuffle import shuffle


class ShuffleNetV2Block(chainer.link.Chain):

    def __init__(self, in_channels, out_channels,
                 stride=1, splits_left=2, initialW=None):
        super(ShuffleNetV2Block, self).__init__()

        with self.init_scope():
            if stride == 2:
                self.conv1 = L.Convolution2D(
                    in_channels, in_channels, 1, 1, 0, initialW=initialW,
                    nobias=True)
                self.bn1 = L.BatchNormalization(in_channels)
                self.conv2 = L.DepthwiseConvolution2D(
                    in_channels, 1, 3, stride, 1,
                    initialW=initialW, nobias=True)
                self.bn2 = L.BatchNormalization(in_channels)
                self.conv3 = L.Convolution2D(
                    in_channels, out_channels // 2, 1, 1, 0, initialW=initialW,
                    nobias=True)
                self.bn3 = L.BatchNormalization(out_channels // 2)
                self.conv4 = L.DepthwiseConvolution2D(
                    in_channels, 1, 3, stride, 1,
                    initialW=initialW, nobias=True)
                self.bn4 = L.BatchNormalization(in_channels)
                self.conv5 = L.Convolution2D(
                    in_channels, out_channels // 2, 1, 1, 0, initialW=initialW,
                    nobias=True)
                self.bn5 = L.BatchNormalization(out_channels // 2)
            elif stride == 1:
                self.in_channels = in_channels - in_channels // splits_left
                self.conv1 = L.Convolution2D(
                    self.in_channels, self.in_channels, 1, 1, 0,
                    initialW=initialW, nobias=True)
                self.bn1 = L.BatchNormalization(self.in_channels)
                self.conv2 = L.DepthwiseConvolution2D(
                    self.in_channels, 1, 3, stride, 1,
                    initialW=initialW, nobias=True)
                self.bn2 = L.BatchNormalization(self.in_channels)
                self.conv3 = L.Convolution2D(
                    self.in_channels, self.in_channels, 1, 1, 0,
                    initialW=initialW, nobias=True)
                self.bn3 = L.BatchNormalization(self.in_channels)
            self.stride = stride
            self.splits_left = splits_left

    def __call__(self, x):
        if self.stride == 2:
            h1 = F.relu(self.bn1(self.conv1(x)))
            h1 = self.bn2(self.conv2(h1))
            h1 = F.relu(self.bn3(self.conv3(h1)))

            h2 = self.bn4(self.conv4(x))
            h2 = F.relu(self.bn5(self.conv5(h2)))
        elif self.stride == 1:
            h1, h2 = F.split_axis(x, (self.in_channels,), axis=1)
            h1 = F.relu(self.bn1(self.conv1(h1)))
            h1 = self.bn2(self.conv2(h1))
            h1 = F.relu(self.bn3(self.conv3(h1)))

        h = F.concat((h1, h2))
        return shuffle(h)


class BuildingShuffleNetV2Block(chainer.link.Chain):

    def __init__(self, n_layer, in_channels,
                 out_channels, stride, splits_left, initialW=None):
        super(BuildingShuffleNetV2Block, self).__init__()
        with self.init_scope():
            self.a = ShuffleNetV2Block(
                in_channels, out_channels, stride, splits_left, initialW)
            self._forward = ["a"]
            for i in range(n_layer - 1):
                name = 'b{}'.format(i + 1)
                bottleneck = ShuffleNetV2Block(out_channels, out_channels, 1,
                                               splits_left, initialW)
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


class ShuffleNetV2(chainer.link.Chain):

    def __init__(self, n_layers, n_out, net_scale=1.0, splits_left=2,
                 layer_names=None):
        super().__init__()
        kwargs = {'initialW': normal.HeNormal(scale=1.0)}

        if n_layers == 50:
            block = [3, 4, 6, 3]
        elif n_layers == 164:
            block = [10, 10, 23, 10]
        else:
            raise ValueError('The n_layers argument should be either 50,'
                             ' or 164, but {} was given.'.format(n_layers))

        if net_scale == 0.5:
            out_channels = [24, 48, 96, 192, 1024]
        elif net_scale == 1.0:
            out_channels = [24, 116, 232, 464, 1024]
        elif net_scale == 1.5:
            out_channels = [24, 176, 352, 704, 1024]
        elif net_scale == 2.0:
            out_channels = [24, 244, 488, 976, 2048]
        else:
            raise ValueError('net_scale augment should be either 0.5, 1.0,'
                             ' 1.5, or 2.0, but {} was given.'.format(
                                 net_scale))

        with self.init_scope():
            self.conv1 = L.Convolution2D(3, out_channels[0], 3, 1, 1,
                                         nobias=True, **kwargs)
            self.bn1 = L.BatchNormalization(out_channels[0])
            self.res2 = BuildingShuffleNetV2Block(block[0], out_channels[0],
                                                  out_channels[1], 2,
                                                  splits_left, **kwargs)
            self.res3 = BuildingShuffleNetV2Block(block[1], out_channels[1],
                                                  out_channels[2], 2,
                                                  splits_left, **kwargs)
            self.res4 = BuildingShuffleNetV2Block(block[2], out_channels[2],
                                                  out_channels[3], 2,
                                                  splits_left, **kwargs)
            self.conv5 = L.Convolution2D(out_channels[3], out_channels[4],
                                         1, 1, 0, nobias=True, **kwargs)
            self.bn5 = L.BatchNormalization(out_channels[4])
            self.fc6 = L.Linear(out_channels[4], n_out)

        self.functions = collections.OrderedDict([
            ('conv1', [self.conv1, self.bn1, F.relu]),
            ('pool1', [lambda x: F.max_pooling_2d(x, ksize=3, stride=2)]),
            ('res2', [self.res2]),
            ('res3', [self.res3]),
            ('res4', [self.res4]),
            ('conv5', [self.conv5, self.bn5, F.relu]),
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
        return chainer.cuda.to_cpu(self(x).data)
