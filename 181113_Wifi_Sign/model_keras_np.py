import os
import tensorflow as tf
import numpy as np
import pandas as pd
import pickle
import gzip
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

# data path
path_csi =  'J:\\Data\\Wi-Fi_processed\\'
path_csi_hc = 'J:\\Data\\Wi-Fi_HC\\180_100\\'
path_np = 'C:\\tmp\\'

# data info
df_info = pd.read_csv('data_subc_sig.csv')
# parameters
no_classes = np.max(df_info['id_person'])
epochs = 10
batch_size = 10
#image_height = int(np.max(df_info['len']))
image_height,image_width = 500,30
input_shape = (image_height, image_width, 6)

# data info
df_info = pd.read_csv('data_subc_sig_v1.csv')
#df_info = df_info[df_info.id_person < 50]

person_uid = np.unique(df_info['id_person'])
csi_uid = np.unique(df_info['id'])
dict_id = dict(zip(person_uid,np.arange(len(person_uid))))

# parameters
max_value = np.max(df_info['max'].values)
#no_classes = len(np.unique(df_info['id_person']))
no_classes = len(dict_id)
csi_time = 500 #int(np.max(df_info['len']))
csi_subc = 30
input_shape = (csi_time, csi_subc, 6)

#read file
#file_list = os.listdir(path_csi_hc)
#file = file_list[0]
data_read = np.load(path_np + 'arr_abs.npy')
label_read = np.load(path_np + 'arr_lab.npy')

data_read_f = data_read[:,:500,:,:,:]
label_y = np.array([dict_id[row] for row in label_read[:,0]])

# train test split
idx_tr,idx_te = train_test_split(range(len(data_read_f)),test_size=0.2,random_state=10)

#input data
data_xs_tr = data_read_f[idx_tr].astype('float32') / max_value
data_xs_te = data_read_f[idx_te].astype('float32') / max_value
data_y_tr = label_y[idx_tr].astype('int')
data_y_te = label_y[idx_te].astype('int')

x_train = data_xs_tr.reshape([-1,500,30,6])
x_test = data_xs_te.reshape([-1,500,30,6])
y_train =  tf.keras.utils.to_categorical(data_y_tr, no_classes)
y_test =  tf.keras.utils.to_categorical(data_y_te, no_classes)

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

# training
simple_cnn_model = simple_cnn(input_shape)
simple_cnn_model.fit(x_train, y_train, batch_size, epochs, (x_test, y_test))

train_loss, train_accuracy = simple_cnn_model.evaluate(
    x_train, y_train, verbose=0)
print('Train data loss:', train_loss)
print('Train data accuracy:', train_accuracy)

# test
test_loss, test_accuracy = simple_cnn_model.evaluate(
    x_test, y_test, verbose=0)
print('Test data loss:', test_loss)
print('Test data accuracy:', test_accuracy)