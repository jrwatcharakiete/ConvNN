def flatten(convplane):
    channels, h, w = convplane.shape
    flat = convplane.reshape((channels*h*w, 1))

    return flat
