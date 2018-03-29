def pgp(x, r=2):
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
