import weakref
import collections

import chainer
from chainer import cuda
import chainer.functions as F
import chainer.links as L
import chainer.links.model.vision.resnet as R
try:
    import cupy
    fuse = cupy.fuse
except ImportError:
    fuse = lambda *_, **__: lambda f: f


class InplaceConcatenator(object):

    """Large storage to concatenate feature maps in place."""

    def __init__(self, full_size):
        self._full_size = full_size
        self._storage = None
        self._sizes = []

    def forward(self, to, x):
        return InplaceConcat(self)(to, x)

    def alloc(self, x):
        xp = cuda.get_array_module(x.array)
        shape = list(x.shape)
        shape[1] = self._full_size
        self._storage = xp.empty(shape, dtype=x.dtype)

        new_array = self._storage[:, :x.shape[1]]
        new_array[...] = x.array
        x.array = new_array

    def concat(self, to, x):
        assert to.base is self._storage
        old_channels = to.shape[1]
        new_channels = old_channels + x.shape[1]
        new = self._storage[:, :new_channels]
        new[:, old_channels:] = x
        return new


class InplaceConcat(chainer.Function):

    def __init__(self, concatenator):
        self._concatenator = concatenator

    def forward(self, inputs):
        self.retain_inputs(())
        return self._concatenator.concat(*inputs),

    def backward(self, inputs, grad_outputs):
        if len(self.inputs) == 1:
            return grad_outputs
        gy, = grad_outputs
        n_channels_first = self.inputs[0].shape[1]
        return gy[:, :n_channels_first], gy[:, n_channels_first:]


class RecomputedBNReluConv(chainer.Chain):

    def __init__(self, in_channels, out_channels, kernel_size, pad):
        super(RecomputedBNReluConv, self).__init__()
        self.pad = pad
        with self.init_scope():
            self.bn = L.BatchNormalization(in_channels)
            self.conv = L.Convolution2D(
                in_channels, out_channels, kernel_size, 1, pad,
                initialW=chainer.initializers.HeNormal(), nobias=True)

    def __call__(self, x):
        bn_fn = None
        conv_fn = None
        out_size = None

        def forward(x):
            nonlocal bn_fn, conv_fn, out_size

            if not chainer.config.enable_backprop:
                # forget phase
                with chainer.force_backprop_mode():
                    y = self.bn(x)
                bn_fn = y.creator
                bn_fn.unchain()

                y = F.relu(y)

                with chainer.force_backprop_mode():
                    y = self.conv(y)
                conv_fn = y.creator
                conv_fn.unchain()

                out_size = y.shape
                return y

            # recompute bn using computed statistics
            expander = bn_fn.expander
            bn_out = self._recompute_bn(x.array, self.bn.gamma.array[expander],
                                        self.bn.beta.array[expander],
                                        bn_fn.mean[expander],
                                        bn_fn.inv_std[expander])
            bn_out = chainer.Variable(bn_out)
            bn_fn.inputs = x.node, self.bn.gamma.node, self.bn.beta.node
            bn_fn.outputs = weakref.ref(bn_out.node),
            bn_out.creator_node = bn_fn
            x.retain_data()
            self.bn.gamma.retain_data()
            self.bn.beta.retain_data()

            # recompute relu
            h = F.relu(bn_out)

            # set dummy data to convolution output
            xp = cuda.get_array_module(h.array)
            conv_fn.inputs = h.node, self.conv.W.node
            h.retain_data()
            self.conv.W.retain_data()
            dummy_out = chainer.Variable(xp.broadcast_to(
                xp.empty((), dtype=h.dtype), out_size))
            conv_fn.outputs = weakref.ref(dummy_out.node),
            dummy_out.creator_node = conv_fn

            bn_fn = None
            conv_fn = None
            return dummy_out

        return F.forget(forward, x)

    @staticmethod
    @fuse()
    def _recompute_bn(x, gamma, beta, mean, inv_std):
        return (x - mean) * inv_std * gamma + beta


class DenseLayer(chainer.Chain):

    def __init__(self, concatenator, in_ch, growth_rate):
        super(DenseLayer, self).__init__()
        mid_ch = growth_rate * 4
        self.concatenator = concatenator

        with self.init_scope():
            self.l1 = RecomputedBNReluConv(in_ch, mid_ch, 1, 0)
            self.l2 = RecomputedBNReluConv(mid_ch, growth_rate, 3, 1)

    def __call__(self, x):
        h = self.l1(x)
        h = self.l2(h)
        return self.concatenator.forward(x, h)


class TransitionLayer(chainer.Chain):

    def __init__(self, in_ch, out_ch):
        super(TransitionLayer, self).__init__()
        with self.init_scope():
            self.bn = L.BatchNormalization(in_ch)
            self.conv = L.Convolution2D(in_ch, out_ch, 1, 1, 0)

    def __call__(self, x):
        x = self.conv(F.relu(self.bn(x)))
        x = F.average_pooling_2d(x, 2, stride=2)
        return x


class DenseBlock(chainer.Chain):

    def __init__(self, num_layers, in_ch, growth_rate):
        super(DenseBlock, self).__init__()
        self.layers = []
        self.concatenator = InplaceConcatenator(
            in_ch + num_layers * growth_rate)

        ch = in_ch
        with self.init_scope():
            for i in range(num_layers):
                layer = DenseLayer(self.concatenator, ch, growth_rate)
                ch += growth_rate
                setattr(self, 'layer{}'.format(i), layer)
                self.layers.append(layer)

    def __call__(self, x):
        # assume x is a variable
        self.concatenator.alloc(x)
        for layer in self.layers:
            x = layer(x)
        return x


class DenseNetBC(chainer.Chain):

    def __init__(self, n_out, stage_sizes, growth_rate, reduction=0.5,
                 layer_names=None):
        super(DenseNetBC, self).__init__()
        ch = growth_rate * 2
        self.stages = []
        self.n_blocks = len(stage_sizes)

        with self.init_scope():
            self.conv1 = L.Convolution2D(
                3, ch, 3, 1, 1, initialW=chainer.initializers.HeNormal(),
                nobias=True)

            for i, s in enumerate(stage_sizes):
                block = DenseBlock(s, ch, growth_rate)
                ch += s * growth_rate
                setattr(self, 'block{}'.format(i), block)
                self.stages.append(block)

                if i + 1 < len(stage_sizes):
                    ch2 = int(ch * reduction)
                    trans = TransitionLayer(ch, ch2)
                    ch = ch2
                    setattr(self, 'trans{}'.format(i), trans)
                    self.stages.append(trans)

            self.fc_bn = L.BatchNormalization(ch)
            self.fc = L.Linear(ch, n_out)

        self.functions = collections.OrderedDict([
            ('conv1', [self.conv1]),
            ('block2', [self.block0]),
            ('trans2', [self.trans0]),
            ('block3', [self.block1]),
            ('trans3', [self.trans1]),
            ('block4', [self.block2]),
            ('pool4', [self.fc_bn, F.relu, R._global_average_pooling_2d]),
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
