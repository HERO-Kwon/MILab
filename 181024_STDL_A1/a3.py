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
learning_rate = 0.0001
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

# create general iterator
iterator = tf.data.Iterator.from_structure(train_dataset.output_types,
train_dataset.output_shapes)
next_element = iterator.get_next()

# make datasets that we can initialize separately, but using the same structure via the common iterator
training_init_op = iterator.make_initializer(train_dataset)
test_init_op = iterator.make_initializer(test_dataset)

def CNNmodel(in_data):
    input2d = tf.reshape(in_data, [-1,32,32,3])
    conv11 = tf.layers.conv2d(
        inputs=input2d, 
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
    conv22 = tf.layers.conv2d(
        inputs=conv21, 
        filters=64, 
        kernel_size=[3, 3], 
        strides=1,
        padding="same", 
        activation=tf.nn.relu)
    pool2 = tf.layers.max_pooling2d(
        inputs=conv22, 
        pool_size=[2, 2], 
        strides=1)
    conv31 = tf.layers.conv2d(
        inputs=pool2, 
        filters=128, 
        kernel_size=[3, 3], 
        strides=1,
        padding="same", 
        activation=tf.nn.relu)
    conv32 = tf.layers.conv2d(
        inputs=conv31, 
        filters=128, 
        kernel_size=[3, 3], 
        strides=1,
        padding="same", 
        activation=tf.nn.relu)
    conv33 = tf.layers.conv2d(
        inputs=conv32, 
        filters=128, 
        kernel_size=[3, 3], 
        strides=1,
        padding="same", 
        activation=tf.nn.relu)
    pool3 = tf.layers.max_pooling2d(
        inputs=conv33, 
        pool_size=[2, 2], 
        strides=1)
    pool_flat = tf.reshape(pool3, [-1, 29 * 29 * 128])
    fc = tf.layers.dense(
        inputs= pool_flat, units=500)#, activation=tf.nn.relu)
    output = tf.layers.dense(inputs=fc, units=10)    
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
training_init_op = iterator.make_initializer(train_dataset)
test_init_op = iterator.make_initializer(test_dataset)

# run the training
epochs = training_epochs
results = pd.DataFrame(columns=['loss','test_loss','accuracy','test_accuracy'])
result_trainingfc = pd.DataFrame(columns=['fc','labels'])
result_testfc = pd.DataFrame(columns=['fc','labels'])
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


'''
    sess.run(training_init_op)
    for j in range(400):
        training_fc,training_labels = sess.run([fc,labels])
        val_training_labels = [np.argmax(r) for r in training_labels]
        result_trainingfc.loc[j] = [list(training_fc),val_training_labels]
    sess.run(test_init_op)
    for k in range(400):
        test_fc,test_labels = sess.run([fc,labels])
        val_test_labels = [np.argmax(r) for r in test_labels]
        result_testfc.loc[k] = [list(test_fc),val_test_labels]
    
#flower plot
list_trainfc = []
for a in range(len(result_trainingfc)):
    for b in range(training_batch):
        fc1 = result_trainingfc.iloc[a].fc[b][0]
        fc2 = result_trainingfc.iloc[a].fc[b][1]
        lab = result_trainingfc.iloc[a].labels[b]
        list_trainfc.append([fc1,fc2,lab])
plt_trainfc = list(zip(*list_trainfc))
colormap=plt_trainfc[2]
plt.scatter(plt_trainfc[0],plt_trainfc[1],c=colormap,s=0.1)
plt.colorbar()

#flower plot(test)
list_testfc = []
for a in range(len(result_testfc)):
    for b in range(test_batch):
        fc1 = result_testfc.iloc[a].fc[b][0]
        fc2 = result_testfc.iloc[a].fc[b][1]
        lab = result_testfc.iloc[a].labels[b]
        list_testfc.append([fc1,fc2,lab])
plt_testfc = list(zip(*list_testfc))
colormap=plt_testfc[2]
plt.scatter(plt_testfc[0],plt_testfc[1],c=colormap,s=0.1)
plt.colorbar()

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
'''