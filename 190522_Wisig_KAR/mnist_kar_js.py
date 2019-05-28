# library
import numpy as np
import pandas as pd

# keras version

import keras
keras.__version__


#convnet filters
from keras import layers
from keras import models

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


# karnet input

x = train_images
y = train_labels


# make karnet model
outputs = x

# retraining classifier
X_inv = np.linalg.pinv(x)

# Initialize W2, W3
W2 = np.random.rand(1024, 512)
W2_inv = np.linalg.pinv(W2)
W3 = np.random.rand(512, 10)
W3_inv = np.linalg.pinv(W3)

# W1 optimization
W1 = np.dot(y, W3_inv)
W1 = np.arctanh(W1)
W1 = np.dot(W1, W2_inv)
W1 = np.arctanh(W1)
W1 = np.dot(X_inv, W1)

outputs = np.dot(outputs, W1)
outputs = np.tanh(outputs)
outputs_inv = np.linalg.pinv(outputs)

# W2 optimization
W2 = np.dot(y, W3_inv)
W2 = np.arctanh(W2)
W2 = np.dot(outputs_inv, W2)
                
outputs = np.dot(outputs, W2)
outputs = np.tanh(outputs)
outputs_inv = np.linalg.pinv(outputs)

# W3 optimization
W3 = np.dot(outputs_inv, y)


# test

outputs1 = test_images
outputs1 = np.dot(outputs1, W1)
outputs1 = np.tanh(outputs1)
outputs1 = np.dot(outputs1, W2)
outputs1 = np.tanh(outputs1)
outputs1 = np.dot(outputs1, W3)


pred = pd.Series([np.argmax(v) for v in outputs1])
truth = pd.Series([np.argmax(v) for v in test_labels])

#accuracy
acc = np.sum(pred==truth) / len(pred)
print(acc)