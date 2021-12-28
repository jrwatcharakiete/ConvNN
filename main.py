from flatten import *
from train import *
import cv2
import numpy as np

im = cv2.imread('./images/1.jpg')
im = im.reshape(3,177,285) # convert imfile to be (num_chanels, h, w)
fts = np.random.normal(size=(8, 3, 3, 3))
bias = np.zeros((fts.shape[0], 1))
conv_out = convolution(im, fts, bias)
conv_out = maxpooling(conv_out, ft=20, stride=100)
conv_out = flatten(conv_out)
conv_out = conv_out/100
# print(conv_out, conv_out.shape)
#t = np.random.normal((5,1))
t = conv_out
t = softmax(t)
# print(t, t.shape)
proba = t
label = np.zeros((t.shape[0], 1))
label[7] = 1
cost = categorical_crossentropy(proba, label)
#print(f'cost = {cost:.2f}'.format(cost=cost))

train()
