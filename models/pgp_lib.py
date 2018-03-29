import numpy as np
import chainer
import chainer.functions as F


def pgp(x, r=2):
    in_batch, in_channel, in_height, in_width = x.data.shape
    if not in_height % r == 0:
        x = zero_pads(x, in_height % r, 2)
    if not in_width % r == 0:
        x = zero_pads(x, in_width % r, 3)
    in_batch, in_channel, in_height, in_width = x.data.shape
    out_batch, out_channel, out_height, out_width = int(in_batch * (r ** 2)), in_channel, int(in_height / r),  int(in_width / r)
    h = x.reshape(in_batch, in_channel, out_height, r, out_width, r)
    h = h.transpose(3, 5, 0, 1, 2, 4)
    return h.reshape(out_batch, out_channel, out_height, out_width)


def pgp_inv(x, r=2):
    in_batch, in_channel, in_height, in_width = x.data.shape
    out_batch, out_channel, out_height, out_width = int(in_batch / (r ** 2)), in_channel, r * in_height, r * in_width
    h = x.reshape(r, r, out_batch, in_channel, in_height, in_width)
    h = h.transpose(2, 3, 4, 0, 5, 1)
    return h.reshape(out_batch, out_channel, out_height, out_width)


def zero_pads(x, pad, where):
    sizes = list(x.data.shape)
    sizes[where] = pad
    xp = chainer.cuda.get_array_module(x)
    pad_mat = chainer.Variable(chainer.cuda.to_gpu(xp.zeros(sizes, dtype=np.float32), device=chainer.cuda.get_device_from_array(x.data)))
    return F.concat((pad_mat, x), axis=where)
