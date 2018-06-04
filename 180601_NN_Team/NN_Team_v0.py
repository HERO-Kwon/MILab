import numpy as np
import tensorflow as tf
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
data_train = (x_train,y_train)
data_test = (x_test,y_test)

def gen_readnpy(files_data):
    for i in range(len(files_data[0])):
        data_read = np.load(files_data[0][i]).astype('float32')
        label_read = np.load(files_data[1][i]).astype('int32')

        yield data_read,label_read

iter_train = gen_readnpy(data_train)
iter_test = gen_readnpy(data_test)

tf.reset_default_graph()
## Network
learning_rate = 0.001
n_input = 6*30
n_step  = 500
n_class = 104
total_batch = 6
n_hidden = 128
batch_size = 1000

X = tf.placeholder(tf.float32,[None,n_step,n_input])
Y = tf.placeholder(tf.int32,[None,n_class])

W = tf.Variable(tf.random_normal([n_hidden,n_class]))
b = tf.Variable(tf.random_normal([n_class]))

cell = tf.nn.rnn_cell.BasicRNNCell(n_hidden)

outputs, states = tf.nn.dynamic_rnn(cell,X,dtype=tf.float32)

outputs = tf.transpose(outputs,[1,0,2])
outputs = outputs[-1]

model = tf.matmul(outputs,W) + b

cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
    logits=model,labels=Y))

optimizer = tf.train.AdamOptimizer(learning_rate).minimize(cost)

#tf.reset_default_graph()

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    total_cost = 0
    for i in range(total_batch):
        try:
            batch_xs,batch_ys = next(iter_train)
            batch_ys = np.eye(n_class)[batch_ys].squeeze(1)
            batch_xs = batch_xs.reshape((batch_size,n_step,n_input))
            _,cost_val = sess.run([optimizer,cost],
            feed_dict = {X:batch_xs,Y:batch_ys})

            total_cost += cost_val
            print(cost_val)
            
            #value = sess.run(next_element)
            #print(f"{value[1].shape}", end=" ")
        except tf.errors.OutOfRangeError:
            print("End of Dataset")
        #print('Epoch:','%04d' % (epoch+1),
        #'Avg.cost = ',':.3f'.format(total_cost / total_batch))


###
'''
def read_npy_file(filename_data,filename_idx):
    data_read = np.float32(np.load(filename_data.decode()))
    idx_read = np.float32(np.load(filename_idx.decode()))
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

handle = tf.placeholder(tf.string, shape=[])
iterator = tf.data.Iterator.from_string_handle(
    handle, training_dataset.output_types, training_dataset.output_shapes)

next_element = iterator_train.get_next()
iterator_train = train_dataset.make_one_shot_iterator()
iterator_test = test_dataset.make_one_shot_iterator()


## Network
learning_rate = 0.001
n_input = 6*30
n_step  = 500
n_class = 100
total_batch = 6
n_hidden = 128
batch_size = 1000

X = tf.placeholder(tf.float32,[None,n_step,n_input])
Y = tf.placeholder(tf.float32,[None,n_class])

W = tf.Variable(tf.random_normal([n_hidden,n_class]))
b = tf.Variable(tf.random_normal([n_class]))

cell = tf.nn.rnn_cell.BasicRNNCell(n_hidden)

outputs, states = tf.nn.dynamic_rnn(cell,X,dtype=tf.float32)

outputs = tf.transpose(outputs,[1,0,2])
outputs = outputs[-1]

model = tf.matmul(outputs,W) + b

cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
    logits=model,labels=Y))

optimizer = tf.train.AdamOptimizer(learning_rate).minimize(cost)

#tf.reset_default_graph()

with tf.Session() as sess:
    training_handle = sess.run(iterator_train.string_handle())
    sess.run(tf.global_variables_initializer())
    total_cost = 0
    for i in range(total_batch):
        try:
            batch_xs,batch_ys = training_handle
            batch_xs = tf.reshape(batch_xs,(batch_size,n_step,n_input))
            _,cost_val = sess.run([optimizer,cost],
            feed_dict = {X:batch_xs,Y:batch_ys})

            total_cost += cost_val
    
            
            #value = sess.run(next_element)
            #print(f"{value[1].shape}", end=" ")
        except tf.errors.OutOfRangeError:
            print("End of Dataset")
        print('Epoch:','%04d' % (epoch+1),
        'Avg.cost = ',':.3f'.format(total_cost / total_batch))

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
'''