import numpy as np
import os
import scipy.io as sio
import re
import tensorflow as tf

# For Windows
file_path_fs = 'D:\Matlab_Drive\Data\MedAI\MAI_Project1\Train_image_label' # im_org - full sampled
file_path_ksps = 'D:\Matlab_Drive\Data\MedAI\MAI_Project1\Train_k'   # f_im - ksp undersampling
file_path_unds = 'D:\Matlab_Drive\Data\MedAI\MAI_Project1\Train_image_input' # im_u - undersampled



# Read Full Sampled Data
data_fs = {}
file_path = file_path_fs

file_list = os.listdir(file_path)
file_list = [s for s in file_list if ".mat" in s]

for file in file_list:

    os.path.join(file_path,file)
    data_read = sio.loadmat(os.path.join(file_path,file))

    data_arr = data_read['im_org']
    data_lab = re.findall('\d+',file)

    data_fs[data_lab[0]] = data_arr


# Read Under Sampled Data
data_unds = {}
file_path = file_path_unds

file_list = os.listdir(file_path)
file_list = [s for s in file_list if ".mat" in s]

for file in file_list:

    os.path.join(file_path,file)
    data_read = sio.loadmat(os.path.join(file_path,file))

    data_arr = data_read['im_u']
    data_lab = re.findall('\d+',file)

    data_unds[data_lab[0]] = data_arr


# hyper parameters

learning_rate = 0.001
training_epochs = 15
batch_size = 2
arr_size = 256 * 256

class Model:
    
    def __init__(self, sess, name):
        self.sess = sess
        self.name = name
        self._build_net()

    def _build_net(self):
        with tf.variable_scope(self.name):
            # dropout (keep_prob) rate  0.7~0.5 on training, but should be 1
            # for testing
            self.keep_prob = tf.placeholder(tf.float32)

            # input place holders
            self.X = tf.placeholder(tf.float32, shape=(None,256,256))
            self.Y = tf.placeholder(tf.float32, shape=(None,256,256))
            # img 28x28x1 (black/white)
            X_img = tf.reshape(self.X, [-1, 256, 256, 1])
            #self.Y = tf.placeholder(tf.float32, [None, 10])

            # L1 ImgIn shape=(?, 28, 28, 1)
            W1 = tf.Variable(tf.random_normal([3, 3, 1, 32], stddev=0.01))
            #    Conv     -> (?, 28, 28, 32)
            #    Pool     -> (?, 14, 14, 32)
            L1 = tf.nn.conv2d(X_img, W1, strides=[1, 1, 1, 1], padding='SAME')
            L1 = tf.nn.relu(L1)
            L1 = tf.nn.max_pool(L1, ksize=[1, 2, 2, 1],
                                strides=[1, 2, 2, 1], padding='SAME')
            L1 = tf.nn.dropout(L1, keep_prob=self.keep_prob)
            '''
            Tensor("Conv2D:0", shape=(?, 28, 28, 32), dtype=float32)
            Tensor("Relu:0", shape=(?, 28, 28, 32), dtype=float32)
            Tensor("MaxPool:0", shape=(?, 14, 14, 32), dtype=float32)
            Tensor("dropout/mul:0", shape=(?, 14, 14, 32), dtype=float32)
            '''
            # L2 ImgIn shape=(?, 14, 14, 32)
            W2 = tf.Variable(tf.random_normal([3, 3, 32, 64], stddev=0.01))
            #    Conv      ->(?, 14, 14, 64)
            #    Pool      ->(?, 7, 7, 64)
            L2 = tf.nn.conv2d(L1, W2, strides=[1, 1, 1, 1], padding='SAME')
            L2 = tf.nn.relu(L2)
            L2 = tf.nn.max_pool(L2, ksize=[1, 2, 2, 1],
                                strides=[1, 2, 2, 1], padding='SAME')
            L2 = tf.nn.dropout(L2, keep_prob=self.keep_prob)
            '''
            Tensor("Conv2D_1:0", shape=(?, 14, 14, 64), dtype=float32)
            Tensor("Relu_1:0", shape=(?, 14, 14, 64), dtype=float32)
            Tensor("MaxPool_1:0", shape=(?, 7, 7, 64), dtype=float32)
            Tensor("dropout_1/mul:0", shape=(?, 7, 7, 64), dtype=float32)
            '''
            # L3 ImgIn shape=(?, 7, 7, 64)
            W3 = tf.Variable(tf.random_normal([3, 3, 64, 128], stddev=0.01))
            #    Conv      ->(?, 7, 7, 128)
            #    Pool      ->(?, 4, 4, 128)
            #    Reshape   ->(?, 4 * 4 * 128) # Flatten them for FC
            L3 = tf.nn.conv2d(L2, W3, strides=[1, 1, 1, 1], padding='SAME')
            L3 = tf.nn.relu(L3)
            L3 = tf.nn.max_pool(L3, ksize=[1, 2, 2, 1], strides=[
                                1, 2, 2, 1], padding='SAME')
            L3 = tf.nn.dropout(L3, keep_prob=self.keep_prob)
            L3_flat = tf.reshape(L3, [-1, 128 * 4 * 4])
            '''
            Tensor("Conv2D_2:0", shape=(?, 7, 7, 128), dtype=float32)
            Tensor("Relu_2:0", shape=(?, 7, 7, 128), dtype=float32)
            Tensor("MaxPool_2:0", shape=(?, 4, 4, 128), dtype=float32)
            Tensor("dropout_2/mul:0", shape=(?, 4, 4, 128), dtype=float32)
            Tensor("Reshape_1:0", shape=(?, 2048), dtype=float32)
            '''

            # L4 FC 4x4x128 inputs -> 625 outputs
            W4 = tf.get_variable("W4", shape=[128 * 4 * 4, 625],
                                 initializer=tf.contrib.layers.xavier_initializer())
            b4 = tf.Variable(tf.random_normal([625]))
            L4 = tf.nn.relu(tf.matmul(L3_flat, W4) + b4)
            L4 = tf.nn.dropout(L4, keep_prob=self.keep_prob)
            '''
            Tensor("Relu_3:0", shape=(?, 625), dtype=float32)
            Tensor("dropout_3/mul:0", shape=(?, 625), dtype=float32)
            '''

            # L5 Final FC 625 inputs -> 10 outputs
            W5 = tf.get_variable("W5", shape=[625, 256*256],
                                 initializer=tf.contrib.layers.xavier_initializer())
            b5 = tf.Variable(tf.random_normal([256*256]))
            self.logits = tf.matmul(L4, W5) + b5
            pred_Y = tf.reshape(self.logits, [-1, 256, 256, 1])
            

            '''
            Tensor("add_1:0", shape=(?, 10), dtype=float32)
            '''
        
        # define cost/loss & optimizer
        self.cost = tf.nn.l2_loss(pred_Y - self.Y)
        self.optimizer = tf.train.AdamOptimizer(
            learning_rate=learning_rate).minimize(self.cost)

        #correct_prediction = tf.equal(
        #    tf.argmax(self.logits, 1), tf.argmax(self.Y, 1))
        self.accuracy = self.cost

    def predict(self, x_test, keep_prop=1.0):
        return self.sess.run(self.logits, feed_dict={self.X: x_test, self.keep_prob: keep_prop})

    def get_accuracy(self, x_test, y_test, keep_prop=1.0):
        return self.sess.run(self.accuracy, feed_dict={self.X: x_test, self.Y: y_test, self.keep_prob: keep_prop})

    def train(self, x_data, y_data, keep_prop=0.7):
        return self.sess.run([self.cost, self.optimizer], feed_dict={
            self.X: x_data, self.Y: y_data, self.keep_prob: keep_prop})

    def temp(self, x_data, y_data, keep_prop=0.7):
        return self.sess.run([self.X, self.Y], feed_dict={
            self.X: x_data, self.Y: y_data, self.keep_prob: keep_prop})

# generator
def GenMiniBatch(dict_train,batch_size):
    key_train = list(dict_train.keys())
    for i in range(batch_size):
        keys = key_train[batch_size * i : batch_size * (i+1) ]
        dict_mini = {k : dict_train[k] for k in keys}
        arr_mini = np.array(list(dict_mini.values()))
        arr_keys = np.array(keys)
        yield(arr_mini,arr_keys)

# initialize
sess = tf.Session()
m1 = Model(sess, "m1")

sess.run(tf.global_variables_initializer())
print('Learning Started!')

# train my model
for epoch in range(training_epochs):
    avg_cost = 0
    #total_batch = int(mnist.train.num_examples / batch_size)
    total_batch = int(np.floor(len(data_unds) / batch_size))

    for i in range(total_batch):
        gen = GenMiniBatch(data_unds,batch_size)
        batch_xs, label_xs = next(gen)

        dict_ys = {k : data_fs[k] for k in label_xs}
        batch_ys = np.array(list(dict_ys.values()))
        c,_ = m1.train(batch_xs, batch_ys)
        #avg_cost += c / total_batch

    #print('Epoch:', '%04d' % (epoch + 1), 'cost =', '{:.9f}'.format(avg_cost))
print('Learning Finished!')
sess.close()
# Test model and check accuracy
#print('Accuracy:', m1.get_accuracy(mnist.test.images, mnist.test.labels))