from chainer.backends import cuda
from chainer import function_node
from chainer.utils import type_check


def _shuffle(x, stride=2, axis=1, inv=False):
    if x.shape[axis] % stride > 0:
        raise ValueError('len(axis){} cannot be devided by stride{}'.format(
            x.shape[axis], stride))
    sh0, sh1 = stride, x.shape[axis] // stride
    if inv:
        sh0, sh1 = sh1, sh0
    shape = x.shape[:axis] + (sh0, sh1) + x.shape[axis + 1:]
    y = x.reshape(shape)
    y = y.swapaxes(axis, axis + 1)
    return y.reshape(*x.shape)


class Shuffle(function_node.FunctionNode):

    """Shuffle function."""

    def __init__(self, stride=2, axis=1, inv=False, mod=0):
        self.stride = stride
        self.axis = axis
        self.inv = inv
        self.given_mod = mod

    def check_type_forward(self, in_types):
        type_check.expect(in_types.size() == 1)
        x_type, = in_types
        if self.axis < 0:
            type_check.expect(x_type.ndim >= -self.axis)
        else:
            type_check.expect(x_type.ndim > self.axis)

    def forward(self, inputs):
        self.retain_inputs(())
        x, = inputs
        self.mod = -x.shape[self.axis] % self.stride
        if self.mod > 0:
            xp = cuda.get_array_module(x)
            pad = [[0, 0]] * len(x.shape)
            pad[self.axis] = [0, self.mod]
            x = xp.pad(x, pad, 'constant')
        y = _shuffle(x, self.stride, self.axis, self.inv)
        if self.given_mod == 0:
            return y,
        return y[((slice(None),) * self.axis) + (slice(-self.given_mod),)],

    def backward(self, indexes, grad_outputs):
        g, = grad_outputs
        gx, = Shuffle(
            self.stride, self.axis, not self.inv, self.mod).apply((g,))
        return gx,


def shuffle(x, stride=2, axis=1, inv=False):
    """Shuffle a given variable along an axis.

    This function shuffle ``x`` with ``stride`` along ``axis``.
    When `x.shape[axis] % stride != 0`, zero-padding is used for fill a shape
    mismatch.

    When ``inv`` is ``True``, inverse shuffle is used.

    Args:
        x (:class:`~chainer.Variable` or :class:`numpy.ndarray` or \
        :class:`cupy.ndarray`):
            Variable to shuffle.
            A :math:`(s_1, s_2, ..., s_N)` -shaped float array.
        stride (:class:`int`): Stride of shuffle application.
        axis (:class:`int`): Axis that the input array is shuffled along.
        inv (:class:`bool`): If ``True``, inverse shuffle is used.

    Returns:
        ~chainer.Variable: Output variable.

    .. admonition: Example

    >>> import numpy as np
    >>> x = np.arange(8).reshape((2, 4)).astype(np.float32)
    >>> x
    array([[0., 1., 2., 3.],
           [4., 5., 6., 7.]], dtype=float32)
    >>> y = shuffle(x)
    >>> y.data
    array([[0., 2., 1., 3.],
           [4., 6., 5., 7.]], dtype=float32)
    >>> y = shuffle(x, inv=True)
    >>> y.data
    array([[0., 2., 1., 3.],
           [4., 6., 5., 7.]], dtype=float32)
    >>> x = np.arange(6).reshape((2, 3)).astype(np.float32)
    >>> x
    array([[0., 1., 2.],
           [3., 4., 5.]], dtype=float32)
    >>> y = shuffle(x)
    >>> y.data
    array([[0., 2., 1., 0.],
           [3., 5., 4., 0.]], dtype=float32)
    """
    y, = Shuffle(stride, axis, inv).apply((x,))
    return y
