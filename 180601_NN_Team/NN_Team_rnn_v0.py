import numpy as np
import tensorflow as tf
import os
from sklearn.model_selection import train_test_split

try:
# For Windows
    file_path = 'D:\\Matlab_Drive\\Data\\WIFI\\180_100_DCout'
    file_list = os.listdir(file_path)
except:
# For Linux
    file_path = '~/data/WIFI/180_100_DCout'
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
batch_size = 1000
total_epoch = 6

n_input = 6*30
n_step  = 500
n_class = 104
total_batch = 6
n_hidden = 128


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

            batch_xs = batch_xs.reshape((batch_size,n_step,n_input))
            batch_ys = np.eye(n_class)[batch_ys].squeeze(1)
            
            _,cost_val = sess.run([optimizer,cost],
            feed_dict = {X:batch_xs,Y:batch_ys})

            total_cost += cost_val
        except tf.errors.OutOfRangeError:
            print("End of Dataset")
        
        print('Epoch:','%04d' % (i+1),
        'Avg.cost = ','{:.3f}'.format(total_cost / total_batch))

