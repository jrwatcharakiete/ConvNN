import numpy as np

def convolution(convplane, ft, bias, stride=1):
    num_fts, ft_channels, ft_h, ft_w = ft.shape
    conv_channels, conv_h, conv_w = convplane.shape


    out_dim_h = int((conv_h-ft_h)/stride)+1
    out_dim_w = int((conv_w-ft_w)/stride)+1
    assert conv_channels == ft_channels, \
        "Dimensions of filter must match dimensions of input image"
    out = np.zeros((num_fts, out_dim_h, out_dim_w))

    for curr_ft in range(num_fts):
        curr_h = out_h = 0
        while curr_h+ft_h <= conv_h:
            curr_w = out_w = 0
            while curr_w+ft_w <= conv_w:
                # * is elementwise multiplication
                out[curr_ft, out_h, out_w] = \
                np.sum(ft[curr_ft]*convplane[:, curr_h:curr_h+ft_h, curr_w:curr_w+ft_w]) + bias[curr_ft]
                curr_w += stride
                out_w += 1
            curr_h += stride
            out_h += 1

    return out
