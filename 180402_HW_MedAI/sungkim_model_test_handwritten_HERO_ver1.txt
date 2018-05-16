# Lab 10 MNIST and NN
import tensorflow as tf
import random
# import matplotlib.pyplot as plt

from tensorflow.examples.tutorials.mnist import input_data

tf.set_random_seed(777)  # reproducibility

mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
# Check out https://www.tensorflow.org/get_started/mnist/beginners for
# more information about the mnist dataset

# parameters
learning_rate = 0.001
training_epochs = 15
batch_size = 100

# input place holders
X = tf.placeholder(tf.float32, [None, 784])
Y = tf.placeholder(tf.float32, [None, 10])

# weights & bias for nn layers
W1 = tf.Variable(tf.random_normal([784, 256]))
b1 = tf.Variable(tf.random_normal([256]))
L1 = tf.nn.relu(tf.matmul(X, W1) + b1)

W2 = tf.Variable(tf.random_normal([256, 256]))
b2 = tf.Variable(tf.random_normal([256]))
L2 = tf.nn.relu(tf.matmul(L1, W2) + b2)

W3 = tf.Variable(tf.random_normal([256, 10]))
b3 = tf.Variable(tf.random_normal([10]))
hypothesis = tf.matmul(L2, W3) + b3

# define cost/loss & optimizer
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
    logits=hypothesis, labels=Y))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

# initialize
sess = tf.Session()
sess.run(tf.global_variables_initializer())

# train my model
for epoch in range(training_epochs):
    avg_cost = 0
    total_batch = int(mnist.train.num_examples / batch_size)

    for i in range(total_batch):
        batch_xs, batch_ys = mnist.train.next_batch(batch_size)
        feed_dict = {X: batch_xs, Y: batch_ys}
        c, _ = sess.run([cost, optimizer], feed_dict=feed_dict)
        avg_cost += c / total_batch

    print('Epoch:', '%04d' % (epoch + 1), 'cost =', '{:.9f}'.format(avg_cost))

print('Learning Finished!')

# Test model and check accuracy
correct_prediction = tf.equal(tf.argmax(hypothesis, 1), tf.argmax(Y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
print('Accuracy:', sess.run(accuracy, feed_dict={
      X: mnist.test.images, Y: mnist.test.labels}))

# Get one and predict
r = random.randint(0, mnist.test.num_examples - 1)
print("Label: ", sess.run(tf.argmax(mnist.test.labels[r:r + 1], 1)))
print("Prediction: ", sess.run(
    tf.argmax(hypothesis, 1), feed_dict={X: mnist.test.images[r:r + 1]}))

# plt.imshow(mnist.test.images[r:r + 1].
#           reshape(28, 28), cmap='Greys', interpolation='nearest')
# plt.show()

#######################################
### Predict Hand written dataset    ###
### Made By : HERO Kwon             ###
### Date : 2018-04-22               ###
### Version : 0.1                   ###
#######################################


import os
import pandas as pd
import numpy as np
import re

from PIL import Image
import PIL.ImageOps  

from matplotlib import pyplot as plt
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import confusion_matrix

array_len = 28*28
file_path = 'D:\Data\handwrite_me'
file_list = os.listdir(file_path)
file_list = [s for s in file_list if ".jpg" in s]

image_raw = pd.DataFrame(columns = ['image','digit','digit_num'])

for file in file_list:
    image_read = Image.open(os.path.join(file_path,file))
    image_read = image_read.convert('L') # convert image to black and white
    image_read = PIL.ImageOps.invert(image_read) # invert image (black to white)
    image_array = np.array(image_read).reshape(-1,array_len) / 255 # normalize value to 0~1

    [digit,digit_num] = re.findall('\d',file)

    data_read = {'image':image_array,'digit':digit,'digit_num':digit_num}
    image_raw = image_raw.append(data_read,ignore_index=True)

# One hot encoding
enc.fit(np.array(range(0,10)).reshape(-1,1)) 
onehot_digit = enc.transform(image_raw.digit.values.reshape(-1,1)).toarray()



# predict MNIST test set
tf_list_mnist = []
for i in range(len(image_raw)):
    r = random.randint(0, mnist.test.num_examples - 1)

    label_mnist = sess.run(tf.argmax(mnist.test.labels[r:r + 1], 1))
    pred_mnist = sess.run(tf.argmax(hypothesis, 1), feed_dict={X: mnist.test.images[r:r + 1]})
    #print("Label / Prediction :  ", label_mnist , "/",pred_mnist)

    tf_list_mnist.append(label_mnist == pred_mnist)

acc_mnist = np.sum(tf_list_mnist) / len(tf_list_mnist)
print("MNIST Test Set Accuracy : ",acc_mnist)

# predict hand written test set
tf_list_hand = []
for i in range(len(image_raw)):

    label_hand = sess.run(tf.argmax(onehot_digit[i].reshape(1,10), 1))
    pred_hand = sess.run(tf.argmax(hypothesis, 1), feed_dict={X: image_raw.image[i]})
    #print("Label / Prediction :  ", label_hand , "/",pred_hand)

    tf_list_hand.append(label_hand==pred_hand)

acc_hand = np.sum(tf_list_hand) / len(tf_list_hand)
print("Hand Written Accuracy : ",acc_hand)