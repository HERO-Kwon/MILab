### For : NN HW #1
### Made by : HERO Kwon
### Date : 20180418


import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
from scipy.stats import truncnorm
from sklearn.model_selection import train_test_split
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
# Target function

x = np.linspace(start=-1.0,stop=1.0,num=1000)
y = np.linspace(start=-1.0,stop=1.0,num=1000)

x_mesh,y_mesh = np.meshgrid(x,y)

f_xy = np.sin(20*np.sqrt(x_mesh**2 + y_mesh**2))/(20*np.sqrt(x_mesh**2+y_mesh**2)) + 1/5*np.cos(10*np.sqrt(x_mesh**2+y_mesh**2)) + y/2-0.3
f_target = f_xy + truncnorm(a=0, b=1).rvs(size=f_xy.shape)

xind_train,xind_test = train_test_split(range(0,1000,1),test_size = 0.3)#,random_state=10)
xind_test,xind_valid = train_test_split(xind_test,test_size = 0.5)#,random_state=10)

yind_train,yind_test = train_test_split(range(0,1000,1),test_size = 0.3)#,random_state=10)
yind_test,yind_valid = train_test_split(yind_test,test_size = 0.5)#,random_state=10)

# Build Model

# hyper parameters

class Model:

    def __init__(self, sess, name, learning_rate, n_hidden):
        self.sess = sess
        self.name = name
        self.n_hidden = n_hidden
        self.learning_rate = learning_rate
        self._build_net()

    def _build_net(self):
        with tf.variable_scope(self.name):
            self.learning_rate = tf.convert_to_tensor(self.learning_rate,dtype=tf.float32)
            # Network Parameters
            n_hidden_1 = self.n_hidden # 1st layer number of neurons
            n_input = 700

            # input place holders
            self.t = tf.placeholder(tf.float32,(),name="init")
            self.X = tf.placeholder(tf.float32,[1,None])
            self.Y = tf.placeholder(tf.float32,[1,None])
            self.f_XY = tf.placeholder(tf.float32,[1,None])

            weights = {
                'w_h1' : tf.Variable(tf.random_normal([n_input,2,n_hidden_1])),
                'w_out' : tf.Variable(tf.random_normal([n_hidden_1,n_input]))
            }
            biases = {
                'b_h1' : tf.Variable(tf.random_normal([n_hidden_1])),
                'b_out' : tf.Variable(tf.random_normal([n_input]))
            }
            
            input_x = tf.expand_dims(tf.reshape(self.X,[-1,1]),2)
            input_y = tf.expand_dims(tf.reshape(self.Y,[-1,1]),2)

            input_stack = tf.concat([input_x,input_y],-1)
            layer_1 = tf.nn.sigmoid(tf.matmul(input_stack, weights['w_h1']) + biases['b_h1'])
            layer_out = tf.nn.sigmoid(tf.matmul(tf.squeeze(layer_1,axis=1), weights['w_out']) + biases['b_out'])

        # define cost/loss & optimizer
        self.learning_rate = tf.multiply(self.learning_rate , self.t)
        self.cost = tf.nn.l2_loss(layer_out-self.f_XY)
        self.optimizer = tf.train.MomentumOptimizer(learning_rate=self.learning_rate,momentum=0.8).minimize(self.cost)

        self.accuracy = self.cost

    def predict(self, t,x_test, y_test):
        return self.sess.run(self.cost, feed_dict={self.t:t,self.X: x_test,self.Y:y_test})

    def get_accuracy(self, t,x_test, y_test, fxy_test):
        return self.sess.run(self.accuracy, feed_dict={self.t:t,self.X: x_test, self.Y: y_test, self.f_XY: fxy_test})

    def train(self, t, x_data, y_data, fxy_data):
        return self.sess.run([self.cost, self.learning_rate, self.optimizer], feed_dict={self.t:t,self.X: x_data, self.Y: y_data, self.f_XY : fxy_data})

# initialize

n_hidden = 3

sess = tf.Session()
m1 = Model(sess, "m1",learning_rate = 0.1,n_hidden = n_hidden)
sess.run(tf.global_variables_initializer())

print('Learning Started!')

# train my model
for i in range(0,10):
    batch_xs = x[xind_train].reshape(1,-1)
    batch_ys = y[yind_train].reshape(1,-1)
    batch_fxys = f_target[xind_train,yind_train].reshape(1,-1)
    t = np.exp(-1*(i))
    cost, rate,_ = m1.train(t ,batch_xs, batch_ys,batch_fxys)

    print('iter =',i+1, 'cost =', '{:.9f}'.format(cost))

print('Learning Finished!')

# Test model and check accuracy

xtest = x[xind_valid].reshape(1,-1)
ytest = y[yind_valid].reshape(1,-1)
fxytest = f_target[xind_valid,yind_valid].reshape(1,-1)

batch_xtest = np.tile(xtest,5)[:,0:700]
batch_ytest = np.tile(ytest,5)[:,0:700]
batch_fxytest = np.tile(fxytest,5)[:,0:700]

print('Test Accuracy:', m1.get_accuracy(t,batch_xtest,batch_ytest,batch_fxytest))
print('Train Accuracy:', m1.get_accuracy(t,batch_xs,batch_ys,batch_fxys))