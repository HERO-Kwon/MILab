import numpy as np
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense
import os
from sklearn.model_selection import train_test_split


# For Windows
file_path = 'D:\\Matlab_Drive\\Data\\WIFI\\180_100_DCout'

# For Linux
# file_path = '/home/hero/Matlab_Drive/Data/ORLDB'

file_list = os.listdir(file_path)
file_list_data = [os.path.join(file_path,s) for s in file_list if not "_idx.npy" in s]
file_list_idx = [os.path.join(file_path,s) for s in file_list if "_idx.npy" in s]



x_train,x_test,y_train,y_test = train_test_split(file_list_data,file_list_idx,test_size=0.2,random_state=33)

with np.load()




###

def read_npy_file(filename_data,filename_idx):
    data_read = np.load(filename_data.decode())
    idx_read = np.load(filename_idx.decode())
    return data_read,idx_read

def make_dataset(filenames_data,filenames_idx):
    dataset = tf.data.Dataset.from_tensor_slices((filenames_data,filenames_idx))
    dataset = dataset.map(
        lambda filename_data,filename_idx: tuple(tf.py_func(
            read_npy_file,[filename_data,filename_idx],[tf.double,tf.double]
        ))
    )
    return dataset

train_dataset = make_dataset(x_train,y_train)
test_dataset = make_dataset(x_test,y_test)

iterator_train = train_dataset.make_one_shot_iterator()
iterator_test = test_dataset.make_one_shot_iterator()
next_element = iterator_train.get_next()

with tf.Session() as sess:
    for i in range(10):
        try:
            value = sess.run(next_element)
            print(f"{value[1].shape}", end=" ")
        except tf.errors.OutOfRangeError:
            print("End of Dataset")



###


# define filename queue
filename_queue = tf.train.string_input_producer(x_train,shuffle=False,name='filename_queue')

# define reader
reader = tf.WholeFileReader()
key,value = reader.read(filename_queue)


with tf.Session() as sess:
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)

    for i in range(100):
        print(sess.run([id, num, year, rtype , rtime]))
    coord.request_stop()
    coord.join(threads) 



file_list = os.listdir(file_path)
file_list_data = [s for s in file_list if not "_idx.npy" in s]
file_list_idx = [s for s in file_list if "_idx.npy" in s]

val_wifi = []
idx_wifi = []

# read data
for file in file_list_data:
    file_wpath = os.path.join(file_path,file)

    data_read = np.load(file_wpath)
    val_wifi.append(data_read)

# read idx
for file in file_list_data:
    file_wpath = os.path.join(file_path,file)

    data_read = np.load(file_wpath)
    val_wifi.append(data_read)

# Train-Test Split

image_train = pd.DataFrame()
image_test = pd.DataFrame()

for person in image_raw.person.unique():
    data_train, data_test = train_test_split(image_raw[image_raw.person == person],test_size = 0.5)
    image_train = image_train.append(data_train)
    image_test = image_test.append(data_test)
