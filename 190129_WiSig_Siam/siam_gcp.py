# MIT image
# Made by : HERO Kwon
# Date : 190108

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import imageio
import skimage.transform

#Functions
def DenoisePCA(target_xs1,num_eig):
    # remove static
    H_mat = target_xs1 / np.mean(target_xs1,axis=0)
    H_mat = np.nan_to_num(H_mat)

    # eig decomposition
    corr_mat = H_mat.T.dot(H_mat)
    eig_v,eig_w = np.linalg.eig(corr_mat)

    # return num_eig vectors
    return(H_mat.dot(eig_w[1:num_eig+1,:].T))


# data path_mi
#path_csi = 'D:\\Data\\Wi-Fi_processed_sm\\'
#path_csi_np = 'D:\\Data\\Wi-Fi_processed_npy\\'
path_meta = '/home/herokwon/mount_data/Data/Wi-Fi_meta/'
#path_sc = 'D:\\Data\\Wi-Fi_info\\'
#path_mit_image = 'G:\\Data\\Wi-Fi_11_mithc'
path_csi_hc = '/home/herokwon/mount_data/Data/Wi-Fi_HC/180_100/'
path_sig2d = '/home/herokwon/mount_data/Data/sig2d_processed/'


#parameters
target_dir = 1
target_loc = 1
img_shape = (150,250)
pca_num = 50



import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

#mnist_data = input_data.read_data_sets('MNIST_data', one_hot=True)

input_size = img_shape[0]*img_shape[1]
no_classes = 108
batch_size = 81
total_batches = 50


def add_variable_summary(tf_variable, summary_name):
  with tf.name_scope(summary_name + '_summary'):
    mean = tf.reduce_mean(tf_variable)
    tf.summary.scalar('Mean', mean)
    with tf.name_scope('standard_deviation'):
        standard_deviation = tf.sqrt(tf.reduce_mean(
            tf.square(tf_variable - mean)))
    tf.summary.scalar('StandardDeviation', standard_deviation)
    tf.summary.scalar('Maximum', tf.reduce_max(tf_variable))
    tf.summary.scalar('Minimum', tf.reduce_min(tf_variable))
    tf.summary.histogram('Histogram', tf_variable)

def add_variable_summary(tf_variable, summary_name):
  with tf.name_scope(summary_name + '_summary'):
    mean = tf.reduce_mean(tf_variable)
    tf.summary.scalar('Mean', mean)
    with tf.name_scope('standard_deviation'):
        standard_deviation = tf.sqrt(tf.reduce_mean(
            tf.square(tf_variable - mean)))
    tf.summary.scalar('StandardDeviation', standard_deviation)
    tf.summary.scalar('Maximum', tf.reduce_max(tf_variable))
    tf.summary.scalar('Minimum', tf.reduce_min(tf_variable))
    tf.summary.histogram('Histogram', tf_variable)
      

def convolution_layer(input_layer, filters, kernel_size=[3, 3],
                      activation=tf.nn.relu):
    layer = tf.layers.conv2d(
        inputs=input_layer,
        filters=filters,
        kernel_size=kernel_size,
        activation=activation
    )
    add_variable_summary(layer, 'convolution')
    return layer


def pooling_layer(input_layer, pool_size=[2, 2], strides=2):
    layer = tf.layers.max_pooling2d(
        inputs=input_layer,
        pool_size=pool_size,
        strides=strides
    )
    add_variable_summary(layer, 'pooling')
    return layer


def dense_layer(input_layer, units, activation=tf.nn.relu):
    layer = tf.layers.dense(
        inputs=input_layer,
        units=units,
        activation=activation
    )
    add_variable_summary(layer, 'dense')
    return layer


def get_model(input_):
    input_reshape = tf.reshape(input_, [-1, 28, 28, 1],
                                 name='input_reshape')
    convolution_layer_1 = convolution_layer(input_reshape, 64)
    pooling_layer_1 = pooling_layer(convolution_layer_1)
    convolution_layer_2 = convolution_layer(pooling_layer_1, 128)
    pooling_layer_2 = pooling_layer(convolution_layer_2)
    flattened_pool = tf.reshape(pooling_layer_2, [-1, 5 * 5 * 128],
                                name='flattened_pool')
    dense_layer_bottleneck = dense_layer(flattened_pool, 1024)
    return dense_layer_bottleneck

left_input = tf.placeholder(tf.float32, shape=[None, input_size])
right_input = tf.placeholder(tf.float32, shape=[None, input_size])
y_input = tf.placeholder(tf.float32, shape=[None, no_classes])
left_bottleneck = get_model(left_input)
right_bottleneck = get_model(right_input)
dense_layer_bottleneck = tf.concat([left_bottleneck, right_bottleneck], 1)
dropout_bool = tf.placeholder(tf.bool)
dropout_layer = tf.layers.dropout(
        inputs=dense_layer_bottleneck,
        rate=0.4,
        training=dropout_bool
    )
logits = dense_layer(dropout_layer, no_classes)

with tf.name_scope('loss'):
    softmax_cross_entropy = tf.nn.softmax_cross_entropy_with_logits_v2(
        labels=y_input, logits=logits)
    loss_operation = tf.reduce_mean(softmax_cross_entropy, name='loss')
    tf.summary.scalar('loss', loss_operation)

with tf.name_scope('optimiser'):
    optimiser = tf.train.AdamOptimizer().minimize(loss_operation)


with tf.name_scope('accuracy'):
    with tf.name_scope('correct_prediction'):
        predictions = tf.argmax(logits, 1)
        correct_predictions = tf.equal(predictions, tf.argmax(y_input, 1))
    with tf.name_scope('accuracy'):
        accuracy_operation = tf.reduce_mean(
            tf.cast(correct_predictions, tf.float32))
tf.summary.scalar('accuracy', accuracy_operation)

session = tf.Session()
session.run(tf.global_variables_initializer())

merged_summary_operation = tf.summary.merge_all()
train_summary_writer = tf.summary.FileWriter('/tmp/train', session.graph)
test_summary_writer = tf.summary.FileWriter('/tmp/test')

# load data, label
path_file = path_csi_hc
target_data = np.load(path_file + 'Dataset_' + str(target_loc) + '_' + str(target_dir)+'.npy')
target_xs = target_data[:,4:].astype('float32').reshape([-1,500,30*6])
label_csi = target_data[:,:4].astype('int')

# PCA Denoise
pca_xs = np.array([DenoisePCA(target_xs[r],pca_num)  for r in range(target_xs.shape[0])])

# data info
df_info = pd.read_csv(path_meta+'df_info_hc.csv')
target_df = df_info[(df_info.id_location==1) & (df_info.id_direction==1)]


from sklearn.model_selection import train_test_split

idx_tr,idx_te = train_test_split(np.arange(len(target_df)))
df_tr = target_df.iloc[idx_tr]
df_te = target_df.iloc[idx_te]


y_train =  tf.keras.utils.to_categorical(df_tr['uid'].values, no_classes)
y_test =  tf.keras.utils.to_categorical(df_te['uid'].values, no_classes)

for batch_no in range(total_batches):
    #mnist_batch = mnist_data.train.next_batch(batch_size)
    #train_images, train_labels = mnist_batch[0], mnist_batch[1]
    
    t1 = target_df.iloc[batch_size*batch_no:batch_size*(batch_no+1)]
    ys = tf.keras.utils.to_categorical(t1['uid'].values, no_classes)
    
    target_labs = t1[['id_person','id_location','id_direction','id_exp']].values
    
    list_pcas = []
    list_imgs = []
    for i in range(len(target_labs)):
        target_idx = np.all(label_csi==target_labs[i],axis=1)
        target_pca = pca_xs[target_idx].squeeze()
        target_img = imageio.imread(path_sig2d + t1.iloc[0].id_paper + '_1.png')
        list_pcas.append(skimage.transform.resize(target_pca / np.max(np.abs(target_pca)),img_shape,mode='constant'))
        list_imgs.append(skimage.transform.resize(target_img,img_shape,mode='constant'))
    arr_pcas = np.array(list_pcas)
    arr_imgs = np.array(list_imgs)
    
    _, merged_summary = session.run([optimiser, merged_summary_operation],
                                    feed_dict={
        left_input:arr_pcas,
        right_input: arr_imgs,
        y_input: ys,
        dropout_bool: True
    })
    train_summary_writer.add_summary(merged_summary, batch_no)
    if batch_no % 10 == 0:
        merged_summary, _ = session.run([merged_summary_operation,
                                         accuracy_operation], feed_dict={
            left_input: arr_pcas,
            right_input: arr_imgs,
            y_input: ys,
            dropout_bool: False
        })
        test_summary_writer.add_summary(merged_summary, batch_no)


        
