##############
### Assignment : NN
### Made by : HERO Kwon
### Date : 20180418


import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
from scipy.stats import truncnorm
from sklearn.model_selection import train_test_split

# Target function

x = np.linspace(start=-1,stop=1,num=1000)
y = np.linspace(start=-1,stop=1,num=1000)

f_xy = np.sin(20*np.sqrt(x**2 + y**2))/(20*np.sqrt(x**2+y**2)) + 1/5*np.cos(10*np.sqrt(x**2+y**2)) + y/2-0.3
f_target = f_xy + truncnorm(a=0, b=1).rvs(size=1000)

ind_train, ind_test = train_test_split(range(0,1000,1),test_size = 0.3,random_state=100)
ind_test,ind_valid = train_test_split(ind_test,test_size = 0.5,random_state=100)


# Build Model

# hyper parameters
learning_rate = 0.001

class Model:

    def __init__(self, sess, name):
        self.sess = sess
        self.name = name
        self._build_net()
    def _build_net(self):
        with tf.variable_scope(self.name):

            # Network Parameters
            n_hidden_1 = 3 # 1st layer number of neurons
            n_input = 700 # training set size

            # input place holders
            self.X = tf.placeholder(tf.float32,[1,n_input])
            self.Y = tf.placeholder(tf.float32,[1,n_input])
            self.f_XY = tf.placeholder(tf.float32,[1,n_input])

            weights = {
                'h1' : tf.Variable(tf.random_normal([n_input,2,n_hidden_1])),
                'out' : tf.Variable(tf.random_normal([n_hidden_1,n_input]))
            }
            biases = {
                'b1' : tf.Variable(tf.random_normal([n_hidden_1])),
                'out' : tf.Variable(tf.random_normal([n_input]))
            }
            
            input_x = tf.expand_dims(tf.reshape(self.X,[-1,1]),2)
            input_y = tf.expand_dims(tf.reshape(self.Y,[-1,1]),2)

            input_stack = tf.concat([input_x,input_y],-1)
            layer_1 = tf.add(tf.matmul(input_stack, weights['h1']), biases['b1'])
            layer_out = tf.matmul(tf.squeeze(layer_1,axis=1), weights['out']) + biases['out']

        # define cost/loss & optimizer
        self.cost = tf.nn.l2_loss(tf.subtract(self.f_XY,layer_out))
        self.optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(self.cost)

        self.accuracy = self.cost

    def predict(self, x_test, y_test):
        return self.sess.run(self.cost, feed_dict={self.X: x_test,self.Y:y_test})

    def get_accuracy(self, x_test, y_test, fxy_test):
        return self.sess.run(self.accuracy, feed_dict={self.X: x_test, self.Y: y_test, self.f_XY: fxy_test})

    def train(self, x_data, y_data, fxy_data):
        return self.sess.run([self.cost, self.optimizer], feed_dict={
            self.X: x_data, self.Y: y_data, self.f_XY : fxy_data})

# initialize
sess = tf.Session()
m1 = Model(sess, "m1")

sess.run(tf.global_variables_initializer())

print('Learning Started!')

# train my model
for i in range(0,10):
    batch_xs = x[ind_train].reshape(1,-1)
    batch_ys = y[ind_train].reshape(1,-1)
    batch_fxys = f_target[ind_train].reshape(1,-1)
    cost, _ = m1.train(batch_xs, batch_ys,batch_fxys)

    print('cost =', '{:.9f}'.format(cost))

print('Learning Finished!')

# Test model and check accuracy
#print('Accuracy:', m1.get_accuracy(mnist.test.images, mnist.test.labels))