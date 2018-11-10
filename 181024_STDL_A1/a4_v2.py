#######################################
### Made By: HERO Kwon              ###
### Date: 2018/10/29                ###
### Desc:                           ###
###  - STDL Assignment, CIFAR       ###
###  - Referenced http://adventuresinmachinelearning.com/tensorflow-dataset-tutorial/
###  - Used Tensorflow Dataset      ###
#######################################

import tensorflow as tf
import random
from matplotlib import pyplot as plt
import pandas as pd
import numpy as np

# parameters
learning_rate = 0.001
training_epochs = 500
training_batch = 100
test_batch = 20
# Load Dataset
train, test = tf.keras.datasets.cifar10.load_data()
train_x, train_y = train
test_x, test_y = test

#data
dx_train = tf.data.Dataset.from_tensor_slices(train_x.astype('float32'))
dy_train = tf.data.Dataset.from_tensor_slices(train_y).map(lambda z: tf.one_hot(z[0],10))
train_dataset = tf.data.Dataset.zip((dx_train, dy_train)).shuffle(100000).repeat().batch(training_batch)

# do the same operations for the test set
dx_test = tf.data.Dataset.from_tensor_slices(test_x.astype('float32'))
dy_test = tf.data.Dataset.from_tensor_slices(test_y).map(lambda z: tf.one_hot(z[0], 10))
test_dataset = tf.data.Dataset.zip((dx_test, dy_test)).shuffle(100000).repeat().batch(test_batch)

# feed dict
keep_prob = tf.placeholder(tf.float32)
train_phase = tf.placeholder(tf.bool)
'''
# create general iterator
iterator = tf.data.Iterator.from_structure(train_dataset.output_types,
train_dataset.output_shapes)
next_element = iterator.get_next()

# make datasets that we can initialize separately, but using the same structure via the common iterator
training_init_op = iterator.make_initializer(train_dataset)
test_init_op = iterator.make_initializer(test_dataset)
'''
# A feedable iterator to toggle between validation and training dataset
handle = tf.placeholder(tf.string, shape=[])
iterator = tf.data.Iterator.from_string_handle(
handle, train_dataset.output_types, train_dataset.output_shapes)
next_element = iterator.get_next()

train_iterator = train_dataset.make_one_shot_iterator()
test_iterator = test_dataset.make_one_shot_iterator()

def CNNmodel(in_data):
    regularizer = tf.contrib.layers.l1_regularizer(scale=0.1)
    input2d = tf.reshape(in_data, [-1,32,32,3])
    input_bn = tf.contrib.layers.batch_norm(input2d, decay= 0.99, is_training=train_phase,center= True, scale=True, reuse= False,updates_collections=None)
    conv11 = tf.layers.conv2d(
        inputs=input_bn, 
        filters=32, 
        kernel_size=[3, 3], 
        strides=1,
        padding="same", 
        activation=tf.nn.relu)
    conv12 = tf.layers.conv2d(
        inputs=conv11, 
        filters=32, 
        kernel_size=[3, 3], 
        strides=1,
        padding="same", 
        activation=tf.nn.relu)
    pool1 = tf.layers.max_pooling2d(
        inputs=conv12, 
        pool_size=[2, 2], 
        strides=1)
    conv21 = tf.layers.conv2d(
        inputs=pool1, 
        filters=64, 
        kernel_size=[3, 3], 
        strides=1,
        padding="same", 
        activation=tf.nn.relu)
    conv211 = tf.layers.conv2d(
        inputs=conv21, 
        filters=32, 
        kernel_size=[1, 1], 
        strides=1,
        padding="same", 
        activation=None)
    conv22 = tf.layers.conv2d(
        inputs=conv211, 
        filters=64, 
        kernel_size=[3, 3], 
        strides=1,
        padding="same", 
        activation=tf.nn.relu)
    '''
    pool2 = tf.layers.max_pooling2d(
        inputs=conv22, 
        pool_size=[2, 2], 
        strides=1)
    '''
    conv31 = tf.layers.conv2d(
        inputs=conv22, 
        filters=128, 
        kernel_size=[3, 3], 
        strides=1,
        padding="same", 
        activation=tf.nn.relu)
    conv311 = tf.layers.conv2d(
        inputs=conv31, 
        filters=64, 
        kernel_size=[1, 1], 
        strides=1,
        padding="same", 
        activation=tf.nn.relu)
    conv32 = tf.layers.conv2d(
        inputs=conv311, 
        filters=128, 
        kernel_size=[3, 3], 
        strides=1,
        padding="same", 
        activation=tf.nn.relu)
    conv321 = tf.layers.conv2d(
        inputs=conv32, 
        filters=64, 
        kernel_size=[1, 1], 
        strides=1,
        padding="same", 
        activation=tf.nn.relu)
    conv33 = tf.layers.conv2d(
        inputs=conv321, 
        filters=128, 
        kernel_size=[3, 3], 
        strides=1,
        padding="same", 
        activation=tf.nn.relu)
    '''
    pool3 = tf.layers.max_pooling2d(
        inputs=conv33, 
        pool_size=[2, 2], 
        strides=1)
    '''
    pool_flat = tf.layers.flatten(conv33)
    fc = tf.layers.dense(
        inputs= pool_flat, units=500)#, activation=tf.nn.relu)
    fc_drop = tf.nn.dropout(fc, keep_prob) 
    output = tf.layers.dense(inputs=fc_drop, units=10)    
    return output,fc

# define cost/loss & optimizer
logits,fc = CNNmodel(next_element[0])
labels = next_element[1]
loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(
    logits=logits, labels=labels))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss)

# accuracy
pred = tf.argmax(logits,1)
equal = tf.equal(pred,tf.argmax(labels,1))
accuracy = tf.reduce_mean(tf.cast(equal,tf.float32))

#init
init_op = tf.global_variables_initializer()
#init_op = tf.contrib.layers.xavier_initializer()
#training_init_op = iterator.make_initializer(train_dataset)
#test_init_op = iterator.make_initializer(test_dataset)

# run the training
epochs = training_epochs
results = pd.DataFrame(columns=['loss','test_loss','accuracy','test_accuracy'])
result_trainingfc = pd.DataFrame(columns=['fc','labels'])
result_testfc = pd.DataFrame(columns=['fc','labels'])
with tf.Session() as sess:
    sess.run(init_op)
    # The `Iterator.string_handle()` method returns a tensor that can be evaluated
   # and used to feed the `handle` placeholder.
    training_handle = sess.run(train_iterator.string_handle())
    test_handle = sess.run(test_iterator.string_handle())
    for i in range(epochs):
        #sess.run(training_init_op)
        l, _, acc = sess.run([loss, optimizer, accuracy], feed_dict={handle: training_handle,keep_prob:0.7,train_phase:True})
        print("Epoch: {}, loss: {:.3f}, training accuracy: {:.2f}%".format(i+1, l, acc * 100))

        # now setup the validation run
        test_iters = 1
        test_l, test_acc = sess.run([loss, accuracy],feed_dict={handle:test_handle,keep_prob:1.0,train_phase:False})
        print("Epoch: {}, test loss: {:.3f}, test accuracy: {:.2f}%".format(i+1, test_l, test_acc * 100))

        results.loc[i] = [l,test_l,acc,test_acc]