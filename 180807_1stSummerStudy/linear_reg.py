import numpy as np
import matplotlib.pyplot as plt
import h5py
import scipy
from PIL import Image
from scipy import ndimage
#from lr_utils import load_dataset

%matplotlib inline

#lr_utils.py
import h5py
    
    
def load_dataset():
    train_dataset = h5py.File('D:\\Matlab_Drive\\Data\\Study_1stsummer\\datasets\\train_catvnoncat.h5', "r")
    train_set_x_orig = np.array(train_dataset["train_set_x"][:]) # your train set features
    train_set_y_orig = np.array(train_dataset["train_set_y"][:]) # your train set labels

    test_dataset = h5py.File('D:\\Matlab_Drive\\Data\\Study_1stsummer\\datasets\\test_catvnoncat.h5', "r")
    test_set_x_orig = np.array(test_dataset["test_set_x"][:]) # your test set features
    test_set_y_orig = np.array(test_dataset["test_set_y"][:]) # your test set labels

    classes = np.array(test_dataset["list_classes"][:]) # the list of classes
    
    train_set_y_orig = train_set_y_orig.reshape((1, train_set_y_orig.shape[0]))
    test_set_y_orig = test_set_y_orig.reshape((1, test_set_y_orig.shape[0]))
    
    return train_set_x_orig, train_set_y_orig, test_set_x_orig, test_set_y_orig, classes

# Loading the data (cat/non-cat)
train_set_x_orig, train_set_y, test_set_x_orig, test_set_y, classes = load_dataset()

# linear reg

X = train_set_x_orig.reshape(train_set_x_orig.shape[0],-1).T/255
Y = train_set_y
w = np.zeros((X.shape[0],1))
b = 0

m = train_set_x_orig.shape[0]
alpha = 0.01

def sigmoid(x):
  return 1 / (1 + 1/np.exp(x))

costs = []

for i in range(500):
    Z = np.dot(w.T,X)+b
    A = sigmoid(Z)
    cost = -1/m*np.sum(*np.log(A)+(1-Y)*np.log(1-A)) 

    dz = A-Y
    dw = (1/m)*np.dot(X,dz.T)
    db = (1/m)*np.sum(dz)

    w = w - alpha*dw
    b = b - alpha*db
    
    costs.append(cost)
# test

X_test = test_set_x_orig.reshape(test_set_x_orig.shape[0],-1).T/255
y_hat = 1*(sigmoid(np.dot(w.T,X_test)+b) >= 0.5)

acc = 1 - np.sum(np.abs(y_hat - test_set_y)) / X_test.shape[1]
print(acc)
plt.plot(np.array(costs))