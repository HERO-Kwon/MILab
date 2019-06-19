import numpy as np

# import mnist data

from keras.datasets import mnist
from keras.utils import to_categorical

(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

train_images = train_images.reshape((60000, 28*28,))
train_images = train_images.astype('float32') / 255

test_images = test_images.reshape((10000, 28*28,))
test_images = test_images.astype('float32') / 255

train_labels = to_categorical(train_labels)
test_labels = to_categorical(test_labels)


#karnet parameters
f = ['atan','atan','atan','atan'] #% try setting H=[60000,8000,300] and slope = [1,1,1,1]
finv = ['tan','tan','tan','tan']  #% try setting H=[60000,8000,300] and slope = [1,1,1,1]
h = [2000,1000,100]
slope = [1,1,1,1]
rseed=8

x = train_images
y = train_labels
xte = test_images
yte = test_labels

bias = np.ones([x.shape[0],1])
X = np.hstack([bias,x])
nrow,ncol = X.shape
slopeinv = [1/slope[i] for i in range(len(slope))]

#f:atan, finv:tan


#initialize
np.random.seed(rseed)
w,v = [],[]
for m in np.arange(len(h)):
    w_row = h[len(h)-1-m]
    if m==0:
        w_col = y.shape[1]
    else:
        w_col = h[len(h)-m]
    
    w.append(np.random.rand(w_row,w_col))
    v.append(np.random.rand(1,w_col))   
w.reverse()
v.reverse()

# equation
def suffix_eq(y,bias,v,w):
    return (np.tan(y)-1*bias.dot(v)).dot(1*np.linalg.pinv(w))

def prefix_eq(X,bias,W):
    return 1*(np.hstack([bias,np.arctan(1*X.dot(W))]))

# get W
W = []

W1 = np.linalg.pinv(X).dot(np.tan((suffix_eq(suffix_eq(suffix_eq(y,bias,v[2],w[2]),bias,v[1],w[1]),bias,v[0],w[0]))))
W.append(W1)

W2 = np.linalg.pinv(prefix_eq(X,bias,W[0])).dot(np.tan(suffix_eq(suffix_eq(y,bias,v[2],w[2]),bias,v[1],w[1])))
W.append(W2)

W3 = np.linalg.pinv(prefix_eq(prefix_eq(X,bias,W[0]),bias,W[1])).dot(np.tan(suffix_eq(y,bias,v[2],w[2])))
W.append(W3)

W4 = np.linalg.pinv(prefix_eq(prefix_eq(prefix_eq(X,bias,W[0]),bias,W[1]),bias,W[2])).dot(np.tan(y))
W.append(W4)


# calc output
def kar_calc(x,W):
    bias = np.ones([x.shape[0],1])
    x_calc = x
    for i in range(len(W)):
        x_calc = 1*np.arctan(np.hstack([bias,x_calc]).dot(W[i]))
    return(x_calc)

#prediction
pred_yt = kar_calc(xte,W)
pred_ytr = kar_calc(x,W)

tf_yt = [np.argmax(pred_yt[i])==np.argmax(yte[i]) for i in range(len(pred_yt))]
tf_ytr = [np.argmax(pred_ytr[i])==np.argmax(y[i]) for i in range(len(pred_ytr))]

acc_t = np.sum(tf_yt) / len(tf_yt)
acc_tr = np.sum(tf_ytr) / len(tf_ytr)