import os
import tensorflow as tf
import numpy as np
import pandas as pd
import pickle
import gzip
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

# data path
path_csi_hc = '/home/herokwon/mount_data/Data/Wi-Fi_HC/180_100/'
path_csi_np = '/home/herokwon/mount_data/Data/wisig_np/'
path_meta = '/home/herokwon/mount_data/Data/Wi-Fi_meta/'
path_csi_hdfs = 'J:\\Data\\Wi-Fi_Processed_HDFS\\'

# data info
df_info = pd.read_csv(path_meta + 'data_subc_sig_v1.csv')
# parameters
no_classes = np.max(df_info['id_person'])
epochs = 20
batch_size = 50
#image_height = int(np.max(df_info['len']))
image_height,image_width = 500,30
input_shape = (image_height, image_width, 6)

#read file
file_list = os.listdir(path_csi_np)
file = file_list[0]
data_read = np.load(path_csi_np + file)        

#filter outliers
xs_med = np.median(np.abs(data_read))
xs_std = np.std(np.abs(data_read))
xs_th = xs_med+xs_std

data_read_f =  np.abs(data_read).reshape([-1,500,30,6]).astype('float32')
data_norm = np.zeros(data_read_f.shape)
for i in range(data_read_f.shape[0]):
    for j in range(data_read_f.shape[2]):
        for k in range(data_read_f.shape[3]):
            data_norm[i,:,j,k] = data_read_f[i,:,j,k] - np.mean(data_read_f[i,:,j,k])
            #data_norm[i,:,j,k] = np.append(np.diff(data_read_f[i,:,j,k]),0)


#Read label
label_data = np.load(path_csi_np + file_list[1])[:,0]
uniq_label = np.unique(label_data)
label_table = pd.Series(np.arange(len(uniq_label)),index=uniq_label)
data_y = np.array([label_table[num] for num in label_data])
no_classes = len(np.unique(data_y))

# train test split
idx_tr,idx_te = train_test_split(np.arange(len(data_read_f)),test_size=0.2,random_state=10)
#input data
data_xs_tr = data_norm[idx_tr] 
data_xs_te = data_norm[idx_te] 
data_y_tr = data_y[idx_tr].astype('int')
data_y_te = data_y[idx_te].astype('int')


#x_train = data_xs_tr.reshape([-1,500,30,6])
#x_test = data_xs_te.reshape([-1,500,30,6])
x_train = data_xs_tr / xs_th
x_test = data_xs_te / xs_th
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
