import numpy as np
from convolution import *

def convback(dconv_prev, conv_in, ft, stride):
    out_channels, in_channels, ft_h, ft_w = ft.shape
    conv_channels, conv_h, conv_w = conv_in.shape

    dout = np.zeros(conv_in.shape)
    dft = np.zeros(ft.shape)
    dbias = np.zeros((out_channels, 1))
    for channel in range(out_channels):
        curr_h = out_h = 0
        while curr_h + ft_h <= conv_h:
            curr_w = out_w = 0
            while curr_w + ft_w <= conv_w:
                dft[channel] += dconv_prev[channel, out_h, out_w]*conv_in[:, curr_h:curr_h+ft_h, curr_w:curr_w+ft_w]
                dout[:, curr_h:curr_h+ft_h, curr_w:curr_w+ft_w] += dconv_prev[channel, out_h, out_w]*ft[channel]
                curr_w += stride
                out_w += 1
            curr_h += stride
            out_h += 1
        dbias[channel] = np.sum(dconv_prev[channel])

    return  dout, dft, dbias

def convolutionback(dconv_prev, conv_in, ft, stride):
    num_fts, ft_channels, ft_h, ft_w = ft.shape
    conv_channels, conv_h, conv_w = conv_in.shape

    dout = np.zeros(conv_in.shape)
    dft = np.zeros(ft.shape)
    dbias = np.zeros((num_fts, 1))

    _, dconv_prev_h, dconv_prev_w = dconv_prev.shape

    dconv_prev_4dim = np.reshape(dconv_prev, (1, dconv_prev.shape[0], dconv_prev.shape[1], dconv_prev.shape[2]))
    #print(conv_in.shape, dconv_prev.shape)
    #print(dft.shape)
    # c = convolution(conv_in, dconv_prev_4dim,[0], stride)
    # print(c, c.shape)

    for curr_ft in range(num_fts):
        curr_h = out_h = 0
        while curr_h+ft_h <= conv_h:
            curr_w = out_w = 0
            while curr_w+ft_w <= conv_w:
                dft[curr_ft] += dconv_prev[curr_ft, out_h, out_w] * conv_in[:, curr_h:curr_h+ft_h, curr_w:curr_w+ft_w]
                dout[:, curr_h:curr_h+ft_h, curr_w:curr_w+ft_w] += dconv_prev[curr_ft, out_h, out_w]*ft[curr_ft]
                # dft[curr_ft, out_h, out_w] = \
                #     np.sum(dconv_prev[curr_ft] * conv_in[curr_ft, curr_h:curr_h+dconv_prev_h,
                #                                  curr_w:curr_w+dconv_prev_w])

                curr_w += stride
                out_w += 1
            curr_h += stride
            out_h += 1
        dbias[curr_ft] = np.sum(dconv_prev[curr_ft])

    return dout, dft, dbias

# calculate the index of max (not absolute max) in the  matrix with idxs = tuple
# corresponding to matrix shape
def nanargmax(arr):
    idx = np.nanargmax(arr)
    idxs = np.unravel_index(idx, arr.shape)
    return idxs


def maxpoolBackward(dpool, orig, f, s):
    (n_c, orig_dim, _) = orig.shape

    dout = np.zeros(orig.shape)

    for curr_c in range(n_c):
        curr_y = out_y = 0
        while curr_y + f <= orig_dim:
            curr_x = out_x = 0
            while curr_x + f <= orig_dim:
                # obtain index of largest value in input for current window
                (a, b) = nanargmax(orig[curr_c, curr_y:curr_y + f, curr_x:curr_x + f])
                dout[curr_c, curr_y + a, curr_x + b] = dpool[curr_c, out_y, out_x]

                curr_x += s
                out_x += 1
            curr_y += s
            out_y += 1

    return dout

