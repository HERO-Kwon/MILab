### For : NN HW #1
### Made by : HERO Kwon
### Date : 20180418

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

            # input place holders
            self.t = tf.placeholder(tf.float32,(),name="init")
            self.X = tf.placeholder(tf.float32,[None,1],name="X")
            self.Y = tf.placeholder(tf.float32,[None,1],name="Y")
            self.f_XY = tf.placeholder(tf.float32,[None,1],name="f_XY")

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
        self.learning_rate = tf.multiply(self.learning_rate , self.t)
        self.fxy_hat = layer_out
        self.cost = tf.norm(layer_out-self.f_XY)
        self.optimizer = tf.train.MomentumOptimizer(learning_rate=self.learning_rate,momentum=0.8).minimize(self.cost)
        #self.optimizer = tf.train.AdamOptimizer().minimize(self.cost)
        self.accuracy = self.cost

    def predict(self, t,x_test, y_test):
        return self.sess.run(self.fxy_hat, feed_dict={self.t:t,self.X: x_test,self.Y:y_test})

    def get_accuracy(self, t,x_test, y_test, fxy_test):
        return self.sess.run(self.accuracy, feed_dict={self.t:t,self.X: x_test, self.Y: y_test, self.f_XY: fxy_test})

    def train(self, t, x_data, y_data, fxy_data):
        return self.sess.run([self.cost, self.learning_rate, self.optimizer], feed_dict={self.t:t,self.X: x_data, self.Y: y_data, self.f_XY : fxy_data})

# Main
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
#f_target = f_xy + truncnorm(a=0, b=1).rvs(size=f_xy.shape)
f_target = f_xy + np.random.normal(size=f_xy.shape)


# for HW # 1

def run_model(n_hidden_val):
    xind_train,xind_test = train_test_split(range(0,1000,1),test_size = 0.3)#,random_state=10)
    xind_test,xind_valid = train_test_split(xind_test,test_size = 0.5)#,random_state=10)

    yind_train,yind_test = train_test_split(range(0,1000,1),test_size = 0.3)#,random_state=10)
    yind_test,yind_valid = train_test_split(yind_test,test_size = 0.5)#,random_state=10)

    # initialize
    n_hidden = n_hidden_val
    sess = tf.Session()
    m1 = Model(sess, "m1",learning_rate = 0.1,n_hidden = n_hidden)
    
    sess.run(tf.global_variables_initializer())

    #print('Learning Started!')

    # train my model
    for i in range(0,10):
        batch_xs = x[xind_train].reshape(-1,1)
        batch_ys = y[yind_train].reshape(-1,1)
        batch_fxys = f_target[xind_train,yind_train].reshape(-1,1)
        t = np.exp(-1*i)
        cost, rate,_ = m1.train(t ,batch_xs, batch_ys,batch_fxys)

        #print('iter =',i+1, 'rate =' '{:.3f}'.format(rate) ,'cost =' '{:.3f}'.format(cost))

    #print('Learning Finished!')

    # Test model and check accuracy

    batch_xtest = x[xind_valid].reshape(-1,1)
    batch_ytest = y[yind_valid].reshape(-1,1)
    batch_fxytest = f_target[xind_valid,yind_valid].reshape(-1,1)

    acc_test = m1.get_accuracy(t,batch_xtest,batch_ytest,batch_fxytest)
    acc_train = m1.get_accuracy(t,batch_xs,batch_ys,batch_fxys)
    #print('Test Accuracy:', acc_test)
    #print('Train Accuracy:', acc_train)

    sess.close()

    return([n_hidden,acc_test,acc_train])



result_hn1 = pd.DataFrame(columns = ['n_hidden','acc_test','acc_valid'])
for iter in range(1,26):
    print('processing : ',iter)
    for j in range(0,100):
        result = run_model(iter)
        res_df = {'n_hidden':result[0],'acc_test':result[1],'acc_valid':result[2]}
        result_hn1 = result_hn1.append(res_df,ignore_index=True)

pd.wite_csv(result_hn1,"result_hn1.csv")    

# for HW # 2
# Run Model
xind_train,xind_test = train_test_split(range(0,1000,1),test_size = 0.3)#,random_state=10)
xind_test,xind_valid = train_test_split(xind_test,test_size = 0.5)#,random_state=10)

yind_train,yind_test = train_test_split(range(0,1000,1),test_size = 0.3)#,random_state=10)
yind_test,yind_valid = train_test_split(yind_test,test_size = 0.5)#,random_state=10)

# initialize
n_hidden = 1
sess = tf.Session()
m1 = Model(sess, "m1",learning_rate = 0.1,n_hidden = n_hidden)

sess.run(tf.global_variables_initializer())

#print('Learning Started!')

# train my model

ma = 5

for i in range(0,10):
    batch_xs = x[xind_train].reshape(-1,1)
    batch_ys = y[yind_train].reshape(-1,1)
    batch_fxys = f_target[xind_train,yind_train].reshape(-1,1)
    batch_fxys_ma = pd.rolling_mean(batch_fxys,3)


    t = np.exp(-1*i)
    cost, rate,_ = m1.train(t ,batch_xs[ma:], batch_ys[ma:],batch_fxys_ma[ma:])

    print('iter =',i+1, 'rate =' '{:.3f}'.format(rate) ,'cost =' '{:.3f}'.format(cost))

#print('Learning Finished!')

# Test model and check accuracy

batch_xtest = x[xind_test].reshape(-1,1)
batch_ytest = y[yind_test].reshape(-1,1)
batch_fxytest = f_target[xind_test,yind_test].reshape(-1,1)

acc_test = m1.get_accuracy(t,batch_xtest,batch_ytest,batch_fxytest)
acc_train = m1.get_accuracy(t,batch_xs[ma:],batch_ys[ma:],batch_fxys_ma[ma:])
#print('Test Accuracy:', acc_test)
#print('Train Accuracy:', acc_train)

pred_fxy = m1.predict(t,batch_xtest,batch_ytest)

sess.close()

# Make the plot
fig = plt.figure()
ax = fig.gca(projection='3d')
# Plot the surface.
surf = ax.plot_surface(y_mesh, x_mesh, f_target, cmap=cm.coolwarm,linewidth=0, antialiased=False)
fig.colorbar(surf, shrink=0.5, aspect=5)
plt.show()


xplot_index = batch_xtest.reshape(1,-1).argsort()
yplot_index = batch_ytest.reshape(1,-1).argsort()

sorted_xplot = batch_xtest[xplot_index]
sorted_yplot = batch_ytest[yplot_index]

#sorted_fxy = f_xy[xind_test].reshape(-1,1)[xplot_index]
sorted_fxy_noise_x = batch_fxytest[xplot_index]
sorted_pred_x = pred_fxy[xplot_index]

sorted_fxy_noise_y = batch_fxytest[yplot_index]
sorted_pred_y = pred_fxy[yplot_index]

plt.plot(sorted_xplot.squeeze(),sorted_fxy_noise_x.squeeze(),color='gray')
plt.scatter(sorted_xplot.squeeze(),sorted_pred_x.squeeze(),color='red')

plt.show()

plt.plot(sorted_yplot.squeeze(),sorted_fxy_noise_y.squeeze(),color='gray')
plt.scatter(sorted_yplot.squeeze(),sorted_pred_y.squeeze(),color='red')

plt.show()