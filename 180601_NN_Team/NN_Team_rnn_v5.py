import numpy as np
import tensorflow as tf
import os
from sklearn.model_selection import train_test_split

#os.environ["CUDA_VISIBLE_DEVICES"]="-1"    
try:
# For Windows
    file_path = 'D:\\Matlab_Drive\\Data\\WIFI\\180_100_DCout'
    file_list = os.listdir(file_path)
except:
# For Linux
    file_path = '/home/herokwon/data/WIFI/180_100_DCout'
    file_list = os.listdir(file_path)

file_list_data = [os.path.join(file_path,s) for s in file_list if not "_idx.npy" in s]
file_list_idx = [os.path.join(file_path,s) for s in file_list if "_idx.npy" in s]

#x_train,x_test,y_train,y_test = train_test_split(file_list_data,file_list_idx,test_size=0.2,random_state=33)
#data_train = (x_train,y_train)
#data_test = (x_test,y_test)

def gen_readnpy(files_data,train_size,test_size):
    num_files = len(files_data[0])
    num_minibatch = int(1000 / (train_size + test_size))

    for i in range(num_files):
        data_read = np.load(files_data[0][i]).astype('float32')
        label_read = np.load(files_data[1][i]).astype('int32')

        x_train,x_test,y_train,y_test = train_test_split(data_read,label_read,test_size=0.2,random_state=33)
        
        for j in range(num_minibatch):
            train_x_give = x_train[train_size*j:train_size*(j+1)]
            train_y_give = y_train[train_size*j:train_size*(j+1)]
            test_x_give = x_test[test_size*j:test_size*(j+1)]
            test_y_give = y_test[test_size*j:test_size*(j+1)]

            yield train_x_give,train_y_give,test_x_give,test_y_give

#iter = gen_readnpy((file_list_data,file_list_idx))
#iter_test = gen_readnpy(data_test)

## Network

global_step = tf.Variable(0,trainable=False,name='global_step')

learning_rate = 0.01
train_batch_size = 80
test_batch_size = 20
total_epoch = 3

n_input = 6*30
n_step  = 500
n_class = 110
total_batch = int(8000 / (train_batch_size+test_batch_size))
n_hidden = 1024

X = tf.placeholder(tf.float32,[None,n_step,n_input])
Y = tf.placeholder(tf.float32,[None,n_class])

W = tf.Variable(tf.random_normal([n_hidden,n_class]))
b = tf.Variable(tf.random_normal([n_class]))

cell = tf.nn.rnn_cell.BasicLSTMCell(n_hidden)

outputs, states = tf.nn.dynamic_rnn(cell,X,dtype=tf.float32)

outputs = tf.transpose(outputs,[1,0,2])
outputs = outputs[-1]

model = tf.matmul(outputs,W) + b

cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(
    logits=model,labels=Y))

optimizer = tf.train.AdamOptimizer(learning_rate)
train_op = optimizer.minimize(cost,global_step=global_step)
# Accuracy
is_correct = tf.equal(tf.argmax(model,1),tf.argmax(Y,1))
accuracy = tf.reduce_mean(tf.cast(is_correct,tf.float32))

tf.summary.scalar('cost',cost)
tf.summary.scalar('accuracy',accuracy)
#tf.summary.scalar('accuracy',accuracy)


#tf.reset_default_graph()

sess = tf.Session()
sess.run(tf.global_variables_initializer())
total_cost = 0

merged=tf.summary.merge_all()
writer=tf.summary.FileWriter('./logs',sess.graph)
test_res = []

for epoch in range(total_epoch):
    iter = gen_readnpy((file_list_data,file_list_idx),train_batch_size,test_batch_size)
    for step in range(total_batch):

        batch_xs,batch_ys,batch_xst,batch_yst = next(iter)

        batch_xs = batch_xs.reshape((train_batch_size,n_input,n_step))
        batch_xs = np.transpose(batch_xs,[0,2,1])
        batch_ys = np.eye(n_class)[batch_ys].squeeze(1)
        batch_ys = batch_ys.astype('float32')
        
        sess.run(train_op,feed_dict = {X:batch_xs,Y:batch_ys})
        acc = sess.run(accuracy,feed_dict = {X:batch_xs,Y:batch_ys})
        print('Step: %d, '% sess.run(global_step),
        'Cost: %.3f' % sess.run(cost,feed_dict = {X:batch_xs,Y:batch_ys}))

        summary = sess.run(merged,feed_dict={X:batch_xs,Y:batch_ys})
        writer.add_summary(summary,global_step=sess.run(global_step))
        # Accuracy
        #is_correct = tf.equal(tf.argmax(model,1),tf.argmax(Y,1))
        #accuracy = tf.reduce_mean(tf.cast(is_correct,tf.float32))

        #acc_t = sess.run(accuracy,feed_dict = {X:batch_xst,Y:batch_yst})
        batch_xst = batch_xst.reshape((test_batch_size,n_input,n_step))
        batch_xst = np.transpose(batch_xst,[0,2,1])
        batch_yst = np.eye(n_class)[batch_yst].squeeze(1)
        batch_yst = batch_yst.astype('float32')

        acc_t,cost_val_t = sess.run([accuracy,cost],feed_dict = {X:batch_xst,Y:batch_yst})
        test_res.append([cost_val_t,acc_t])
        print('cost_t = ',cost_val_t,'Acc = ',acc_t)

np.savetxt("test_res.csv", test_res, delimiter=",", fmt='%s')

'''
batch_xst = batch_xst.reshape((test_batch_size,n_input,n_step))
batch_xst = np.transpose(batch_xst,[0,2,1])
batch_yst = np.eye(n_class)[batch_yst].squeeze(1)
batch_yst = batch_yst.astype('float32')

acc_t,cost_val_t = sess.run([accuracy,cost],feed_dict = {X:batch_xst,Y:batch_yst})
print('cost_t = ',cost_val_t,'Acc = ',acc_t)
'''
#!rm -r logs
#!tensorboard --logdir=./logs