import chainer
from chainer.backends import cuda
from chainer import function_node
from chainer.utils import type_check


class PGP(function_node.FunctionNode):

    """Parallel Grid Pooling."""

    def __init__(self, r=2):
        self.r = r

    def check_type_forward(self, in_types):
        type_check.expect(in_types.size() == 1)
        type_check.expect(
            in_types[0].dtype.kind == 'f',
            in_types[0].ndim == 4
        )

    def forward(self, inputs):
        X, = inputs
        xp = cuda.get_array_module(X)
        bsize, c, a, b = X.shape
        if a % self.r:
            X = zero_pads(X, a % self.r, 2)
        if b % self.r:
            X = zero_pads(X, b % self.r, 3)

        bsize, c, a, b = X.shape
        X = xp.reshape(
            X, (bsize, c, a // self.r, self.r, b // self.r, self.r))
        X = xp.transpose(X, (3, 5, 0, 1, 2, 4))
        X = xp.reshape(
            X, (self.r ** 2 * bsize, c, a // self.r, b // self.r))
        return X,

    def backward(self, indexes, grad_outputs):
        gy, = grad_outputs
        bsize, c, a, b = gy.shape
        bsize //= self.r ** 2
        gy = chainer.functions.reshape(gy, (self.r, self.r, bsize, c, a, b))
        gy = chainer.functions.transpose(gy, (2, 3, 4, 0, 5, 1))
        gy = chainer.functions.reshape(gy, (bsize, c, a * self.r, b * self.r))
        return gy,


def pgp(X, r):
    """Parallel grid pooling function.

    Args:
        X (:class:`~chainer.Variable` or :class:`numpy.ndarray` or \
        :class:`cupy.ndarray`):
            Variable holding a 4d array of shape
            ``(batch * r * r, channel, dim1, dim2)``.
        r (int): the downscaling factor.

    Returns:
        ~chainer.Variable:
            A variable holding the downscaled layer array from subpixel array
            sampling. The shape is ``(batch * r * r, channel, dim1, dim2)``

    """
    return PGP(r).apply((X,))[0]


class PGPInverse(function_node.FunctionNode):

    """The inverse of Parallel Grid Pooling."""

    def __init__(self, r=2):
        self.r = r

    def check_type_forward(self, in_types):
        type_check.expect(in_types.size() == 1)
        type_check.expect(
            in_types[0].dtype.kind == 'f',
            in_types[0].ndim == 4
        )

    def forward(self, inputs):
        X, = inputs
        xp = cuda.get_array_module(X)
        bsize, c, a, b = X.shape
        bsize //= self.r ** 2
        X = xp.reshape(X, (self.r, self.r, bsize, c, a, b))
        X = xp.transpose(X, (2, 3, 4, 0, 5, 1))
        X = xp.reshape(X, (bsize, c, a * self.r, b * self.r))
        return X,

    def backward(self, indexes, grad_outputs):
        gy, = grad_outputs
        bsize, c, a, b = gy.shape
        if a % self.r:
            gy = zero_pads(gy, a % self.r, 2)
        if b % self.r:
            gy = zero_pads(gy, b % self.r, 3)

        bsize, c, a, b = gy.shape
        gy = chainer.functions.reshape(
            gy, (bsize, c, a // self.r, self.r, b // self.r, self.r))
        gy = chainer.functions.transpose(gy, (3, 5, 0, 1, 2, 4))
        gy = chainer.functions.reshape(
            gy, (self.r ** 2 * bsize, c, a // self.r, b // self.r))
        return gy,


def pgp_inv(X, r):
    """The inverse of parallel grid pooling function.

    Args:
        X (:class:`~chainer.Variable` or :class:`numpy.ndarray` or \
        :class:`cupy.ndarray`):
            Variable holding a 4d array of shape
            ``(batch * r * r, channel, dim1, dim2)``.
        r (int): the upscaling factor.

    Returns:
        ~chainer.Variable:
            A variable holding the upscaled array from
            interspersed layers. The shape is
            ``(batch, channel, dim1 * r, dim2 * r)``.

    """
    return PGPInverse(r).apply((X,))[0]


def zero_pads(x, pad, where):
    sizes = list(x.shape)
    sizes[where] = pad
    xp = cuda.get_array_module(x)
    pad_mat = xp.zeros(sizes, dtype=x.float32)
    return xp.concatenate((pad_mat, x), axis=where)
