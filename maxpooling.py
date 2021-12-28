import numpy as np

def maxpooling(convplane, ft=2, stride=2):
    conv_channels, conv_h, conv_w = convplane.shape

    # output shape for maxpooled
    h = int((conv_h-ft)/stride)+1
    w = int((conv_w-ft)/stride)+1

    maxpooled = np.zeros((conv_channels, h, w))

    for channel in range(conv_channels):
        curr_h = out_h = 0
        while curr_h + ft <= conv_h:
            curr_w = out_w = 0
            while curr_w + ft <= conv_w:
                maxpooled[channel, out_h, out_w] = \
                np.max(convplane[channel, curr_h:curr_h+ft, \
                       curr_w:curr_w+ft])
                curr_w += stride
                out_w += 1
            curr_h += stride
            out_h += 1
    return maxpooled
