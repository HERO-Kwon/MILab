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

# data info
df_info = pd.read_csv('data_subc_sig_v1.csv')
#df_info = df_info[df_info.id_person < 50]

person_uid = np.unique(df_info['id_person'])
dict_id = dict(zip(person_uid,np.arange(len(person_uid))))

# parameters
max_value = np.max(df_info['max'].values)
#no_classes = len(np.unique(df_info['id_person']))
no_classes = len(dict_id)
csi_time = int(np.max(df_info['len']))
csi_subc = 30
input_shape = (csi_time, csi_subc, 6)

# make data generator
def gen_csi(df_info,id_num,len_num):
    for file in np.unique(df_info.id.values):
        # read sample data
        # load and uncompress.
        with gzip.open(path_csi+file+'.pickle.gz','rb') as f:
            data1 = pickle.load(f)
        '''
        abs_sub = np.mean(np.abs(data1),axis=(0,2,3))
        data1_norm = data1/abs_sub[np.newaxis,:,np.newaxis,np.newaxis]

        data1_abs = np.abs(data1_norm)
        data1_ph = np.angle(data1_norm)
        '''
        data1_diff = np.abs(data1)
        
        # differentiation
        #data1_diff = np.diff(data1_abs,axis=0)
        
        # zero pad
        pad_len = len_num - data1_diff.shape[0]
        data1_pad = np.pad(data1_diff,((0,pad_len),(0,0),(0,0),(0,0)),'constant',constant_values=0)

        # Label
        id_key = df_info[df_info.id==file]['id_person'].values[0].astype('int')
        data1_y = dict_id[id_key]

        x_input = data1_pad.reshape([-1,len_num,30,6]).astype('float32') / max_value
        y_input = tf.keras.utils.to_categorical(data1_y, no_classes).reshape([1,-1]).astype('float32')
        
        yield(x_input ,y_input)
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
    model.add(tf.keras.layers.MaxPooling2D(pool_size=(2,2),strides=10))
    model.add(tf.keras.layers.Dropout(rate=0.3))
    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(units=256, activation='relu'))
    model.add(tf.keras.layers.Dropout(rate=0.3))
    model.add(tf.keras.layers.Dense(units=no_classes, activation='softmax'))
    model.compile(loss=tf.keras.losses.categorical_crossentropy,
                  optimizer=tf.keras.optimizers.Adam(),
                  metrics=['accuracy'])
    return model

#Train Test Split
tr_idx,te_idx = train_test_split(df_info.index,test_size=0.2,random_state=10)
df_train = df_info.loc[tr_idx]
df_test = df_info.loc[te_idx]

gen_train = gen_csi(df_train,no_classes,csi_time)
gen_test = gen_csi(df_test,no_classes,csi_time)

#trining
simple_cnn_model = simple_cnn(input_shape)
simple_cnn_model.fit_generator(gen_train,steps_per_epoch=10,epochs=30,
                              validation_data=gen_test,validation_steps=10)


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