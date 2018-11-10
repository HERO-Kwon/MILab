#######################################
### Made By: HERO Kwon              ###
### Date: 2018/10/24                ###
### Desc:                           ###
###  - STDL Assignment, MNIST       ###
###  - Referenced http://adventuresinmachinelearning.com/tensorflow-dataset-tutorial/
###  - Used Tensorflow Dataset      ###
#######################################

import tensorflow as tf
import random
from matplotlib import pyplot as plt
import pandas as pd


# parameters
learning_rate = 0.001
training_epochs = 100
batch_size = 100

# Load Dataset
train, test = tf.keras.datasets.mnist.load_data()
train_x, train_y = train
test_x, test_y = test

#data
dx_train = tf.data.Dataset.from_tensor_slices(train_x/255)
dy_train = tf.data.Dataset.from_tensor_slices(train_y).map(lambda z: tf.one_hot(z,10))
train_dataset = tf.data.Dataset.zip((dx_train, dy_train)).shuffle(100000).repeat().batch(batch_size)

# do the same operations for the test set
dx_test = tf.data.Dataset.from_tensor_slices(test_x/255)
dy_test = tf.data.Dataset.from_tensor_slices(test_y).map(lambda z: tf.one_hot(z, 10))
test_dataset = tf.data.Dataset.zip((dx_test, dy_test)).shuffle(500).repeat().batch(batch_size)

# create general iterator
iterator = tf.data.Iterator.from_structure(train_dataset.output_types,
train_dataset.output_shapes)
next_element = iterator.get_next()

# make datasets that we can initialize separately, but using the same structure via the common iterator
training_init_op = iterator.make_initializer(train_dataset)
test_init_op = iterator.make_initializer(test_dataset)

def CNNmodel(in_data):
    input2d = tf.reshape(in_data, [-1,28,28,1])
    conv1 = tf.layers.conv2d(
        inputs=input2d, 
        filters=20, 
        kernel_size=[5, 5], 
        strides=1,
        #padding="same", 
        activation=tf.nn.relu)
    pool1 = tf.layers.max_pooling2d(
        inputs=conv1, 
        pool_size=[2, 2], 
        strides=2)
        #padding="same")
    conv2 = tf.layers.conv2d(
        inputs=pool1, 
        filters=50, 
        kernel_size=[5, 5], 
        strides=1,
        #padding="same", 
        activation=tf.nn.relu)
    pool2 = tf.layers.max_pooling2d(
        inputs=conv2, 
        pool_size=[2, 2], 
        strides=2)
        #padding="same")
    pool_flat = tf.reshape(pool2, [-1, 4 * 4 * 50])
    fc = tf.layers.dense(
        inputs= pool_flat, units=500, activation=tf.nn.relu)
    output = tf.layers.dense(inputs=fc, units=10)    
    return output

# define cost/loss & optimizer
logits = CNNmodel(next_element[0])
loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(
    logits=logits, labels=next_element[1]))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss)

# accuracy
pred = tf.argmax(logits,1)
equal = tf.equal(pred,tf.argmax(next_element[1],1))
accuracy = tf.reduce_mean(tf.cast(equal,tf.float32))

#init
init_op = tf.global_variables_initializer()
training_init_op = iterator.make_initializer(train_dataset)
test_init_op = iterator.make_initializer(test_dataset)

# run the training
epochs = training_epochs
results = pd.DataFrame(columns=['loss','test_loss','accuracy','test_accuracy'])
with tf.Session() as sess:
    sess.run(init_op)
    sess.run(training_init_op)
    for i in range(epochs):
        l, _, acc = sess.run([loss, optimizer, accuracy])
        print("Epoch: {}, loss: {:.3f}, training accuracy: {:.2f}%".format(i+1, l, acc * 100))

        # now setup the validation run
        test_iters = 1
        # re-initialize the iterator, but this time with validation data
        sess.run(test_init_op)
        test_l, test_acc = sess.run([loss, accuracy])
        print("Epoch: {}, test loss: {:.3f}, test accuracy: {:.2f}%".format(i+1, test_l, test_acc * 100))

        results.loc[i] = [l,test_l,acc,test_acc]

# loss plot
plt.plot(results.index,results.loss,results.test_loss)
plt.xlim([0,100])
plt.xlabel('Training Iter')
plt.ylabel('Loss')
plt.legend(('Training loss','Test loss'))

# accuracy plot
plt.plot(results.index,results.test_accuracy)
plt.xlim([0,100])
plt.xlabel('Training Iter')
plt.ylabel('Test Accuracy')
#plt.legend('Test Accuracy')
