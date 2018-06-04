import numpy as np
import tensorflow as tf
import os
from sklearn.model_selection import train_test_split

os.environ["CUDA_VISIBLE_DEVICES"]="-1"    
try:
# For Windows
    file_path = 'D:\\Matlab_Drive\\Data\\WIFI\\JS_0312_Final_2'
    file_list = os.listdir(file_path)
except:
# For Linux
    file_path = '/home/herokwon/data/WIFI/JS_0312_Final_2'
    file_list = os.listdir(file_path)

file_list_data = [os.path.join(file_path,s) for s in file_list if not "_idx.npy" in s]
file_list_idx = [os.path.join(file_path,s) for s in file_list if "_idx.npy" in s]

#x_train,x_test,y_train,y_test = train_test_split(file_list_data,file_list_idx,test_size=0.2,random_state=33)
#data_train = (x_train,y_train)
#data_test = (x_test,y_test)

def gen_readnpy(files_data,train_size,test_size):
    for i in range(len(files_data)):
        data_read = np.load(files_data[i]).astype('float32')
        #data_read = np.load(files_data[0][i]).astype('float32')
        #label_read = np.load(files_data[1][i]).astype('int32')
        value_read = data_read[:,1:data_read.shape[1]]
        label_read = data_read[:,0].astype('int32')

        x_train,x_test,y_train,y_test = train_test_split(value_read,label_read,test_size=0.2,random_state=33)
        minibatch_num = int(1000 / (train_size + test_size))
        
        for j in range(minibatch_num):
            train_x_give = x_train[train_size*j:train_size*(j+1)]
            train_y_give = y_train[train_size*j:train_size*(j+1)]
            test_x_give = x_test[test_size*j:test_size*(j+1)]
            test_y_give = y_test[test_size*j:test_size*(j+1)]
            yield train_x_give,train_y_give,test_x_give,test_y_give

#iter = gen_readnpy((file_list_data,file_list_idx))
#iter_test = gen_readnpy(data_test)

tf.reset_default_graph()
## Network
learning_rate = 1
train_batch_size = 80
test_batch_size = 20

n_input = 6
n_step  = 262
n_class = 110
total_batch = int(8000 / (train_batch_size+test_batch_size))
n_hidden = 128

iter = gen_readnpy(file_list_data,train_batch_size,test_batch_size)

X = tf.placeholder(tf.float32,[None,n_step,n_input])
Y = tf.placeholder(tf.int32,[None,n_class])

W = tf.Variable(tf.random_normal([n_hidden,n_class]))
b = tf.Variable(tf.random_normal([n_class]))

cell = tf.contrib.rnn.LayerNormBasicLSTMCell(n_hidden)

outputs, states = tf.nn.dynamic_rnn(cell,X,dtype=tf.float32)

outputs = tf.transpose(outputs,[1,0,2])
outputs = outputs[-1]

model = tf.matmul(outputs,W) + b

cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(
    logits=model,labels=Y))

optimizer = tf.train.AdamOptimizer(learning_rate).minimize(cost)
# Accuracy
is_correct = tf.equal(tf.argmax(model,1),tf.argmax(Y,1))
accuracy = tf.reduce_mean(tf.cast(is_correct,tf.float32))

#tf.reset_default_graph()

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    total_cost = 0

    for i in range(total_batch):
        try:
            batch_xs,batch_ys,batch_xst,batch_yst = next(iter)

            batch_xs = batch_xs.reshape((train_batch_size,n_input,n_step))
            batch_xs = np.transpose(batch_xs,[0,2,1])
            batch_ys = np.eye(n_class)[batch_ys]
            
            _,cost_val,acc_t = sess.run([optimizer,cost,accuracy],feed_dict = {X:batch_xs,Y:batch_ys})
            
            batch_xst = batch_xst.reshape((test_batch_size,n_input,n_step))
            batch_xst = np.transpose(batch_xst,[0,2,1])
            batch_yst = np.eye(n_class)[batch_yst]

            _,cost_val_t = sess.run([accuracy,cost],feed_dict = {X:batch_xst,Y:batch_yst})

            # Accuracy
            #is_correct = tf.equal(tf.argmax(model,1),tf.argmax(Y,1))
            #accuracy = tf.reduce_mean(tf.cast(is_correct,tf.float32))

            #acc_t = sess.run(accuracy,feed_dict = {X:batch_xst,Y:batch_yst})

            total_cost += cost_val

        except tf.errors.OutOfRangeError:
            print("End of Dataset")
        
        print('Epoch:','%04d' % (i+1),
        'Avg.cost = ','{:.3f}'.format(cost_val),'cost_t = ',cost_val_t,'Acc = ',acc_t)
