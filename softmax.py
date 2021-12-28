import numpy as np

def softmax(preds):
    out = np.exp(preds)
    out = out/np.sum(out)
    return out
