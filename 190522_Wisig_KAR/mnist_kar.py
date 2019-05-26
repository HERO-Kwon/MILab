# keras version

import keras
keras.__version__





# import mnist data

from keras.datasets import mnist
from keras.utils import to_categorical

(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

train_images = train_images.reshape((60000, 28*28,))
train_images = train_images.astype('float32') / 255

test_images = test_images.reshape((10000, 28*28,))
test_images = test_images.astype('float32') / 255

train_labels = to_categorical(train_labels)
test_labels = to_categorical(test_labels)



#karnet parameters
f = ['atan','atan','atan','atan'] #% try setting H=[60000,8000,300] and slope = [1,1,1,1]
finv = ['tan','tan','tan','tan']  #% try setting H=[60000,8000,300] and slope = [1,1,1,1]
h = [2000,1000,100]
slope=1
rseed=1
x = train_images
y = train_labels

# make karnet model
import numpy as np
bias = np.ones_like(x[:,0]).reshape((-1,1))

X = np.concatenate([bias,x],axis=1) # add bias to inputs
nrow,ncol = X.shape

#initialize
np.random.seed(rseed)
w,v = [],[]
for m in np.arange(len(h)):
    w_row = h[len(h)-1-m]
    if m==0:
        w_col = y.shape[1]
    else:
        w_col = h[len(h)-m]
    
    w.append(np.random.rand(w_row,w_col))
    v.append(np.random.rand(1,w_col))
w.append(np.zeros((0,0)))
v.append(np.zeros((0,0)))
w.reverse()
v.reverse()

# calc weights
W1 = slope * np.linalg.pinv(X).dot(np.arctan(
    (np.arctan(
        (np.arctan(
            (np.arctan(y)-slope*bias.dot(v[3]))
            .dot(slope*np.linalg.pinv(w[3]))
        )-slope*bias.dot(v[2]))
        .dot(slope*np.linalg.pinv(w[2]))
    )-slope*bias.dot(v[1]))
    .dot(slope*np.linalg.pinv(w[1]))
))

W2 = slope * np.linalg.pinv(np.concatenate([bias,np.tan(slope*X.dot(W1))],axis=1)).dot(
    np.arctan(
        (np.arctan(
            (np.arctan(y)-slope*bias.dot(v[3]))
            .dot(slope*np.linalg.pinv(w[3]))
        )-slope*bias.dot(v[2]))
        .dot(slope*np.linalg.pinv(w[2]))
    ))

W3 = slope * np.linalg.pinv(
    np.concatenate([bias,
                    np.concatenate([bias,np.tan(slope*X.dot(W1))],axis=1).dot(W2)],axis=1)
).dot(
np.arctan(
            (np.arctan(y)-slope*bias.dot(v[3]))
            .dot(slope*np.linalg.pinv(w[3]))
        ))

W4 = slope * np.linalg.pinv(
    np.concatenate([bias,
                    np.concatenate([bias,
                                    np.concatenate([bias,np.tan(slope*X.dot(W1))],axis=1).dot(W2)],axis=1).dot(W3)],axis=1)
).dot(np.arctan(y))


# testing
x = test_images
bias = np.ones_like(x[:,0]).reshape((-1,1))
X = np.concatenate([bias,x],axis=1) # add bias to inputs

netout = np.arctan(slope*
                   np.concatenate([bias,
                                        np.arctan(slope*
                                                  np.concatenate([bias,
                                                                       np.arctan(slope*
                                                                                 np.concatenate([bias,
                                                                                                 np.arctan(slope*X.dot(W1))
                                                                                                      ],axis=1).dot(W2))
                                                                 ],axis=1).dot(W3))
                                  ],axis=1).dot(W4))