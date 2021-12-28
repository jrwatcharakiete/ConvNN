import numpy as np
def categorical_crossentropy(proba, label):
    cost = -np.sum(label*np.log(proba))
    return cost
