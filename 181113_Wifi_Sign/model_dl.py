import os
import tensorflow as tf
import numpy as np
import pandas as pd
import pickle
import gzip

# data path
path_csi =  'J:\\Data\\Wi-Fi_processed\\'

# data info
# data path
path_csi =  'J:\\Data\\Wi-Fi_processed\\'
# data info
df_info = pd.read_csv('data_subc_sig.csv')

# parameters
max_id = np.max(df_info['id_person'])
max_len = int(np.max(df_info['len']))
learning_rate = 0.0001
training_epochs = 600
batch_size = 300

# make data generator
def gen_csi(df_info,id_num,len_num):

    for file in set(df_info.id.values):
        # read sample data
        # load and uncompress.
        with gzip.open(path_csi+file+'.pickle.gz','rb') as f:
            data1 = pickle.load(f)

        # normalize through subc axis
        abs_sub = np.mean(data1_abs,axis=(0,2,3))
        data1_norm = data1/abs_sub[np.newaxis,:,np.newaxis,np.newaxis]

        data1_abs = np.abs(data1_norm)
        data1_ph = np.angle(data1_ph)
        
        # differentiation
        data1_diff = np.diff(data1_abs,axis=0)
        # zero pad
        pad_len = len_num - data1_diff.shape[0]
        data1_pad = np.pad(data1_diff,((0,pad_len),(0,0),(0,0),(0,0)),'constant',constant_values=0)
        # reshape
        data1_resh = data1_pad.reshape(30,max_len,6)
        # Label
        data1_lab = df_info[df_info.id==file]['id_person'].values[0]
        # One hot
        data1_one = np.eye(id_num+1)[data1_lab]
        data1_arrays = [data1_one for one in range(30)]
        data1_stack = np.stack(data1_arrays, axis=0)
        
        for i in range(len(data1_resh)):
            yield(data1_resh[i],data1_stack[i])

# Load Dataset
from sklearn.model_selection import train_test_split
tr_idx,te_idx = train_test_split(df_info.index,test_size=0.2,random_state=10)
#id_num = len(np.unique(df_info.id_person))
df_train = df_info.loc[tr_idx]
df_test = df_info.loc[te_idx]

gen = lambda: (r for r in gen_csi(df_info,max_id,max_len))
train_dataset = tf.data.Dataset().from_generator(gen, (tf.float32,tf.int32)).shuffle(1000).repeat().batch(batch_size)

# create general iterator
iterator = tf.data.Iterator.from_structure(train_dataset.output_types,train_dataset.output_shapes)
next_element = iterator.get_next()

# make datasets that we can initialize separately, but using the same structure via the common iterator
training_init_op = iterator.make_initializer(train_dataset)

def CNNmodel(in_data):
    input1d = tf.reshape(in_data, [-1,max_len,6])
    conv11 = tf.layers.conv1d(
        inputs=input1d, 
        filters=32, 
        kernel_size=3, 
        strides=1,
        padding="same", 
        activation=tf.nn.relu)
    conv12 = tf.layers.conv1d(
        inputs=conv11, 
        filters=32, 
        kernel_size=3, 
        strides=1,
        padding="same", 
        activation=tf.nn.relu)
    pool1 = tf.layers.max_pooling1d(
        inputs=conv12, 
        pool_size=10, 
        strides=10)
    conv21 = tf.layers.conv1d(
        inputs=pool1, 
        filters=64, 
        kernel_size=3, 
        strides=1,
        padding="same", 
        activation=tf.nn.relu)
    conv211 = tf.layers.conv1d(
        inputs=conv21, 
        filters=32, 
        kernel_size=1, 
        strides=1,
        padding="same", 
        activation=None)
    conv22 = tf.layers.conv1d(
        inputs=conv211, 
        filters=64, 
        kernel_size=3, 
        strides=1,
        padding="same", 
        activation=tf.nn.relu)
    
    pool2 = tf.layers.max_pooling1d(
        inputs=conv22, 
        pool_size=10,
        strides=10)
    
    conv31 = tf.layers.conv1d(
        inputs=pool2, 
        filters=128, 
        kernel_size=3, 
        strides=1,
        padding="same", 
        activation=tf.nn.relu)
    conv311 = tf.layers.conv1d(
        inputs=conv31, 
        filters=64, 
        kernel_size=1, 
        strides=1,
        padding="same", 
        activation=tf.nn.relu)
    conv32 = tf.layers.conv1d(
        inputs=conv311, 
        filters=128, 
        kernel_size=3, 
        strides=1,
        padding="same", 
        activation=tf.nn.relu)
    conv321 = tf.layers.conv1d(
        inputs=conv32, 
        filters=64, 
        kernel_size=1, 
        strides=1,
        padding="same", 
        activation=tf.nn.relu)
    conv33 = tf.layers.conv1d(
        inputs=conv321, 
        filters=128, 
        kernel_size=3, 
        strides=1,
        padding="same", 
        activation=tf.nn.relu)
    
    pool3 = tf.layers.max_pooling1d(
        inputs=conv33, 
        pool_size=10, 
        strides=10)
    
    pool_flat = tf.layers.flatten(pool3)
    fc = tf.layers.dense(
        inputs= pool_flat, units=500)#, activation=tf.nn.relu)
    #fc_drop = tf.nn.dropout(fc, keep_prob) 
    output = tf.layers.dense(inputs=fc, units=max_id+1) 
    return output

# define cost/loss & optimizer
logits = CNNmodel(next_element[0])
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
training_init_op = iterator.make_initializer(train_dataset)

# run the training
epochs = training_epochs
results = pd.DataFrame(columns=['loss','accuracy'])
with tf.Session() as sess:
    sess.run(init_op)
    sess.run(training_init_op)
    for i in range(epochs):
        log,lab,pr = sess.run([logits,labels,pred])
        l, _, acc = sess.run([loss, optimizer, accuracy])
        
        print("Epoch: {}, loss: {:.3f}, training accuracy: {:.2f}%".format(i+1, l, acc * 100))
        '''
        # now setup the validation run
        test_iters = 1
        # re-initialize the iterator, but this time with validation data
        sess.run(test_init_op)
        test_l, test_acc = sess.run([loss, accuracy])
        print("Epoch: {}, test loss: {:.3f}, test accuracy: {:.2f}%".format(i+1, test_l, test_acc * 100))
        '''
        results.loc[i] = [l,acc]