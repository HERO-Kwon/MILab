### For : NN HW #1
### Made by : HERO Kwon
### Date : 20180418

# Build Model

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

            # input place holders
            self.t = tf.placeholder(tf.float32,(),name="init")
            self.X = tf.placeholder(tf.float32,[None,1],name="X")
            self.Y = tf.placeholder(tf.float32,[None,1],name="Y")
            self.f_XY = tf.placeholder(tf.float32,[None,1],name="f_XY")

            # xavier Initializer
            initializer = tf.contrib.layers.xavier_initializer()

            weights = {
                'w_h1' : tf.Variable(initializer([2,n_hidden_1])),
                'w_out' : tf.Variable(initializer([n_hidden_1,1]))
            }
            biases = {
                'b_h1' : tf.Variable(initializer([1,n_hidden_1])),
                'b_out' : tf.Variable(initializer([1,1]))
            }

            input_stack = tf.concat([self.X,self.Y],-1)
            layer_1 = tf.nn.sigmoid(tf.matmul(input_stack,weights['w_h1']) + biases['b_h1'])
            layer_out = tf.matmul(layer_1,weights['w_out']) + biases['b_out']

        # define cost/loss & optimizer

        self.learning_rate = tf.multiply(self.learning_rate , self.t) # Modifying Learning Rate
        self.fxy_hat = layer_out # Predicted values
        self.cost = tf.norm(layer_out-self.f_XY) # Used L1 norm for cost function
        self.optimizer = tf.train.MomentumOptimizer\
        (learning_rate=self.learning_rate,momentum=0.8).minimize(self.cost)
        self.accuracy = self.cost

    def predict(self, t,x_test, y_test):
        return self.sess.run(self.fxy_hat, feed_dict={self.t:t,self.X: x_test,self.Y:y_test})

    def get_accuracy(self, t,x_test, y_test, fxy_test):
        return self.sess.run(self.accuracy, \
        feed_dict={self.t:t,self.X: x_test, self.Y: y_test, self.f_XY: fxy_test})

    def train(self, t, x_data, y_data, fxy_data):
        return self.sess.run([self.cost, self.learning_rate, self.optimizer], \
        feed_dict={self.t:t,self.X: x_data, self.Y: y_data, self.f_XY : fxy_data})

# Main
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
from scipy.stats import truncnorm
from sklearn.model_selection import train_test_split
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from scipy import ndimage

# Target function
x = np.linspace(start=-1.0,stop=1.0,num=1000)
y = np.linspace(start=-1.0,stop=1.0,num=1000)

x_mesh,y_mesh = np.meshgrid(x,y)

# Target function
f_xy = np.sin(20*np.sqrt(x_mesh**2 + y_mesh**2))/(20*np.sqrt(x_mesh**2+y_mesh**2)) \
+ 1/5*np.cos(10*np.sqrt(x_mesh**2+y_mesh**2)) + y/2-0.3
f_target = f_xy + np.random.normal(size=f_xy.shape) # Added Gaussian noise
f_target_f = ndimage.uniform_filter(f_target,size=5,mode='constant') # Added average filter for training

# for HW # 1

def run_model(n_hidden_val):

    # Train-Valid-Test split - 7:1.5:1.5
    xind_train,xind_test = train_test_split(range(0,1000,1),test_size = 0.3)
    xind_test,xind_valid = train_test_split(xind_test,test_size = 0.5)

    yind_train,yind_test = train_test_split(range(0,1000,1),test_size = 0.3)
    yind_test,yind_valid = train_test_split(yind_test,test_size = 0.5)

    # initialize
    n_hidden = n_hidden_val # Number of hidden layer
    sess = tf.Session()
    m1 = Model(sess, "m1",learning_rate = 0.1,n_hidden = n_hidden) # make model
    sess.run(tf.global_variables_initializer())

    #print('Learning Started!')
    # train my model
    for i in range(0,10): # 10 iteration for training
        batch_xs = x[xind_train].reshape(-1,1)
        batch_ys = y[yind_train].reshape(-1,1)
        batch_fxys = f_target_f[xind_train,yind_train].reshape(-1,1)
        t = np.exp(-1*i) # decaying learning rate
        cost, rate,_ = m1.train(t ,batch_xs, batch_ys,batch_fxys)

        #print('iter =',i+1, 'rate =' '{:.3f}'.format(rate) ,'cost =' '{:.3f}'.format(cost))
    #print('Learning Finished!')

    # Test model and check accuracy
    batch_xtest = x[xind_valid].reshape(-1,1)
    batch_ytest = y[yind_valid].reshape(-1,1)
    batch_fxytest = f_target[xind_valid,yind_valid].reshape(-1,1)

    # Accuracy 
    acc_test = m1.get_accuracy(t,batch_xtest,batch_ytest,batch_fxytest) 
    acc_train = m1.get_accuracy(t,batch_xs,batch_ys,batch_fxys)
    #print('Test Accuracy:', acc_test)
    #print('Train Accuracy:', acc_train)

    sess.close()
    return([n_hidden,acc_test,acc_train])

result_hn1 = pd.DataFrame(columns = ['n_hidden','acc_test','acc_valid'])
for iter in range(1,21):
    print('processing : ',iter)
    for j in range(0,20):
        result = run_model(iter)
        res_df = {'n_hidden':result[0],'acc_test':result[1],'acc_valid':result[2]}
        result_hn1 = result_hn1.append(res_df,ignore_index=True)

pd.write_csv(result_hn1,"result_hn1.csv")    

# for HW # 2
# Run Model
xind_train,xind_test = train_test_split(range(0,1000,1),test_size = 0.3)#,random_state=10)
xind_test,xind_valid = train_test_split(xind_test,test_size = 0.5)#,random_state=10)

yind_train,yind_test = train_test_split(range(0,1000,1),test_size = 0.3)#,random_state=10)
yind_test,yind_valid = train_test_split(yind_test,test_size = 0.5)#,random_state=10)

# initialize
n_hidden =3
sess = tf.Session()
m1 = Model(sess, "m1",learning_rate = 0.1,n_hidden = n_hidden)

sess.run(tf.global_variables_initializer())

#print('Learning Started!')

# train my model
for i in range(0,10):
    batch_xs = x[xind_train].reshape(-1,1)
    batch_ys = y[yind_train].reshape(-1,1)
    batch_fxys = f_target_f[xind_train,yind_train].reshape(-1,1)
    t = np.exp(-1*(i+1))
    cost, rate,_ = m1.train(t ,batch_xs, batch_ys,batch_fxys)

    #print('iter =',i+1, 'rate =' '{:.3f}'.format(rate) ,'cost =' '{:.3f}'.format(cost))

#print('Learning Finished!')

# Test model and check accuracy

batch_xtest = x[xind_test].reshape(-1,1)
batch_ytest = y[yind_test].reshape(-1,1)
batch_fxytest = f_target[xind_test,yind_test].reshape(-1,1)

acc_test = m1.get_accuracy(t,batch_xtest,batch_ytest,batch_fxytest)
acc_train = m1.get_accuracy(t,batch_xs,batch_ys,batch_fxys)
#print('Test Accuracy:', acc_test)
#print('Train Accuracy:', acc_train)

pred_fxy = m1.predict(t,batch_xtest,batch_ytest)

sess.close()

# For HW #2

# Make the plot
fig = plt.figure()
ax = fig.gca(projection='3d')

# Plot the surface.
surf = ax.plot_surface(y_mesh, x_mesh, f_target, cmap=cm.coolwarm,linewidth=0, antialiased=False)
fig.colorbar(surf, shrink=0.5, aspect=5)
plt.show()

# Sort x,y,fxy array
xplot_index = batch_xtest.reshape(1,-1).argsort()
yplot_index = batch_ytest.reshape(1,-1).argsort()

sorted_xplot = batch_xtest[xplot_index]
sorted_yplot = batch_ytest[yplot_index]

sorted_fxy_noise_x = batch_fxytest[xplot_index]
sorted_pred_x = pred_fxy[xplot_index]
sorted_fxy_noise_y = batch_fxytest[yplot_index]
sorted_pred_y = pred_fxy[yplot_index]

# 2D plot (projection of x,y)

plt.plot(sorted_xplot.squeeze(),sorted_fxy_noise_x.squeeze(),color='gray')
plt.scatter(sorted_xplot.squeeze(),sorted_pred_x.squeeze(),color='red')
plt.show()

plt.plot(sorted_yplot.squeeze(),sorted_fxy_noise_y.squeeze(),color='gray')
plt.scatter(sorted_yplot.squeeze(),sorted_pred_y.squeeze(),color='red')
plt.show()

## For HW #3
import time
# Run Model
xind_train,xind_test = train_test_split(range(0,1000,1),test_size = 0.3)#,random_state=10)
xind_test,xind_valid = train_test_split(xind_test,test_size = 0.5)#,random_state=10)

yind_train,yind_test = train_test_split(range(0,1000,1),test_size = 0.3)#,random_state=10)
yind_test,yind_valid = train_test_split(yind_test,test_size = 0.5)#,random_state=10)

# initialize
n_hidden =3
sess = tf.Session()
m1 = Model(sess, "m1",learning_rate = 0.1,n_hidden = n_hidden)

sess.run(tf.global_variables_initializer())

#result_a3 = pd.DataFrame(columns = ['name','iter','cost','acc_test','time'])

for i in range(0,10):
    batch_xs = x[xind_train].reshape(-1,1)
    batch_ys = y[yind_train].reshape(-1,1)
    batch_fxys = f_target_f[xind_train,yind_train].reshape(-1,1)
    t = np.exp(-1*(i+1))
    t_start = time.time()
    cost, rate,_ = m1.train(t ,batch_xs, batch_ys,batch_fxys)
    t_running = time.time() - t_start

    # Test model and check accuracy
    batch_xtest = x[xind_test].reshape(-1,1)
    batch_ytest = y[yind_test].reshape(-1,1)
    batch_fxytest = f_target[xind_test,yind_test].reshape(-1,1)

    acc_test = m1.get_accuracy(t,batch_xtest,batch_ytest,batch_fxytest)

    res_df = {'name':'sigmoid','iter':i,'cost':cost,'acc_test':acc_test,'time':t_running}
    result_a3 = result_a3.append(res_df,ignore_index=True)


    #print('iter =',i+1, 'rate =' '{:.3f}'.format(rate) ,'cost =' '{:.3f}'.format(cost))

sess.close()
