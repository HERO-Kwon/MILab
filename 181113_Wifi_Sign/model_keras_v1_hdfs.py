import os
import tensorflow as tf
import numpy as np
import pandas as pd
import h5py
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from keras.utils.io_utils import HDF5Matrix

# data path
path_csi =  'J:\\Data\\Wi-Fi_processed\\'
path_csi_hc = 'J:\\Data\\Wi-Fi_HC\\180_100\\'
path_csi_hdfs = 'J:\\Data\\Wi-Fi_Processed_HDFS\\'

# data info
df_info = pd.read_csv('data_subc_sig_v1.csv')
#df_info = df_info[df_info.id_person < 50]

person_uid = np.unique(df_info['id_person'])
dict_id = dict(zip(person_uid,np.arange(len(person_uid))))

# Parameters
max_value = np.max(df_info['max'].values)
no_classes = len(dict_id)
num_csi = 15000
csi_time = 500 #data_csi.shape[1]
csi_subc = 30
input_shape = (csi_time,csi_subc,6)

# nomalization function
def prep_data(array):
    arr_abs = np.abs(array)[:,:csi_time,:,:,:] / max_value
    return arr_abs.reshape([-1,csi_time,csi_subc,6])
def prep_label(array):
    arr_label = array[:,0]
    # Label
    label = np.array([dict_id[arr] for arr in arr_label])
    return tf.keras.utils.to_categorical(label, no_classes)

# Prepare data
x_train = HDF5Matrix(path_csi_hdfs+'csi_label.hdf5','csi',start=0,end=1000,normalizer=prep_data)
x_test = HDF5Matrix(path_csi_hdfs+'csi_label.hdf5','csi',start=1000,end=1100,normalizer=prep_data)
y_train = HDF5Matrix(path_csi_hdfs+'csi_label.hdf5','label',start=0,end=1000,normalizer=prep_label)
y_test = HDF5Matrix(path_csi_hdfs+'csi_label.hdf5','label',start=1000,end=1100,normalizer=prep_label)


#models
def simple_cnn(input_shape):
    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.Conv2D(
        filters=64,
        kernel_size=(3, 3),
        activation='relu',
        input_shape=input_shape
    ))
    model.add(tf.keras.layers.Conv2D(
        filters=128,
        kernel_size=(3, 3),
        activation='relu'
    ))
    model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))
    model.add(tf.keras.layers.Dropout(rate=0.3))
    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(units=256, activation='relu'))
    model.add(tf.keras.layers.Dropout(rate=0.3))
    model.add(tf.keras.layers.Dense(units=no_classes, activation='softmax'))
    model.compile(loss=tf.keras.losses.categorical_crossentropy,
                  optimizer=tf.keras.optimizers.Adam(),
                  metrics=['accuracy'])
    return model

#Fit
simple_cnn_model = simple_cnn(input_shape)
simple_cnn_model.fit(x_train, y_train, epochs=10,batch_size=100,shuffle='batch')

# Plot
def plot_loss(history):
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train','Test'],loc=0)
def plot_acc(history):
    plt.plot(history.history['acc'])
    plt.plot(history.history['val_acc'])
    plt.title('Model Acc')
    plt.ylabel('Acc')
    plt.xlabel('Epoch')
    plt.legend(['Train','Test'],loc=0)

plot_loss(simple_cnn_model.history)
plot_acc(simple_cnn_model.history)


# get accuracy on single data
ex_train = next(gen_train)
train_loss, train_accuracy = simple_cnn_model.evaluate(
    ex_train[0], ex_train[1], verbose=0)
print('Train data loss:', train_loss)
print('Train data accuracy:', train_accuracy)

ex_test = next(gen_test)
test_loss, test_accuracy = simple_cnn_model.evaluate(
    ex_test[0], ex_test[1], verbose=0)
print('Test data loss:', test_loss)
print('Test data accuracy:', test_accuracy)