##########################################
## Made by : HERO Kwon
## Title : TER
## Version : v0
## Date : 2018.05.29.
## Description : TER-RM Algorithm
##########################################


# Main

# packages

import numpy as np
import pandas as pd
import scipy as sp
import os
import re
import imageio
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
import random
from sklearn.metrics import confusion_matrix
import time

from sklearn.datasets import load_iris
from sklearn.model_selection import StratifiedKFold

data = load_iris()
data_v = data['data']
data_l = data['target']

data_train = np.array([]).reshape(0,4)
data_test = np.array([]).reshape(0,4)
label_train = np.array([])
label_test = np.array([])
'''
for target in list(set(data['target'])):
    v_train, v_test = train_test_split(data_v[data_l==target],test_size = 0.5)
    l_train = np.full(shape=len(v_train),fill_value=target)
    l_test = np.full(shape=len(v_test),fill_value=target)
    data_train = np.concatenate((data_train,v_train))
    data_test = np.concatenate((data_test,v_test))
    label_train = np.concatenate((label_train,l_train))
    label_test = np.concatenate((label_test,l_test))
'''

## TER Algorithm
# Basis : RM2
def RMmodel(order,X):
    m,l = X.shape
    
    M1 = []
    M2 = []
    M3 = []
    MM1 = []
    MM3 = []

    Msum = np.sum(X,axis=1)

    for i in range(order):
        for k in range(l):
            M1.append(X[:,k]**(i+1))
            if (i>0):
                M3.append(X[:,k]*Msum**(i)) 
        M2.append(Msum**(i+1))
        MM1.append(M1)
        if (i>0):
            MM3.append(M3)

    MM1 = np.array(MM1).T
    MM1 = MM1.reshape((m,-1,1)).squeeze(axis=2)
    M2 = np.array(M2).T
    if (len(MM3)):
        MM3 = np.array(MM3).T
        MM3 = MM3.reshape((m,-1,1)).squeeze(axis=2)
        P = np.concatenate((np.ones((m,1)),MM1,M2,MM3),axis=1)
    else : P = np.concatenate((np.ones((m,1)),MM1,M2),axis=1)

    return(P)

def TERmodel(rank,r,n,X,Y):
    alpha = []
    for k in list(set(Y)):

        P_n = RMmodel(rank,X[Y!=k])
        P_p = RMmodel(rank,X[Y==k])

        mk_n = X[Y!=k].shape[0]
        mk_p = X[Y==k].shape[0]

        yk_n = (r-n) * np.ones(shape=Y[Y!=k].shape)
        yk_p = (r+n) * np.ones(shape=Y[Y==k].shape)

        I = np.eye(P_n.shape[1])
        b = 10**(-4)

        first_eq = np.linalg.pinv(b*I + (1/mk_n)*(P_n.T).dot(P_n) + (1/mk_p)*(P_p.T).dot(P_p))
        second_eq = (1/mk_n)*(P_n.T).dot(yk_n) + (1/mk_p)*(P_p.T).dot(yk_p)
        ak = np.dot(first_eq,second_eq)

        alpha.append(ak)
    return(np.array(alpha).T)

def TERmodel_new(rank,r,n,X,Y,mod_w):
    #simplified version of TER
    alpha = []
    for k in list(set(Y)):
        P = RMmodel(rank,X)

        mk_n = X[Y!=k].shape[0]
        mk_p = X[Y==k].shape[0]
        w_n = 1/mk_n
        w_p = 1/mk_p

        w_n = mod_w[0][0] * w_n + mod_w[0][1]
        w_p = mod_w[1][0] * w_n + mod_w[1][1]

        ones_mkn = 1*(Y!=k) *w_n
        ones_mkp = 1*(Y==k) *w_p

        W = np.zeros((len(Y), len(Y)), float)
        np.fill_diagonal(W,ones_mkn+ones_mkp)
        yk = ((r-n)*ones_mkn+(r+n)*ones_mkp).T
        ak = np.linalg.pinv((P.T).dot(W).dot(P)).dot(P.T).dot(W).dot(yk)

        alpha.append(ak)
    return(np.array(alpha).T)

M_dict = {}
M_dict['0'] = (0,1),(0,1)
M_dict['1'] = (1,0),(1,0)
M_dict['2'] = (1,0),(0,1)
M_dict['3'] = (0,1),(1,0)
M_dict['4'] = (1,0),(0.5,0)
M_dict['5'] = (0.5,0),(1,0)
M_dict['6'] = (1,0),(0.7,0)
M_dict['7'] = (0.7,0),(1,0)
M_dict['8'] = (1,0),(0.3,0)
M_dict['9'] = (0.3,0),(1,0)


skf = StratifiedKFold(n_splits=10)
acc_MR = pd.DataFrame(columns=['Split','M','R','Acc'])
split = 0

for train_index,test_index in skf.split(data_v,data_l):
    data_train,data_test = data_v[train_index],data_v[test_index]
    label_train,label_test = data_l[train_index],data_l[test_index]
    
    for i in range(10):
        for j in range(10):
            rank = j+1
            alpha = TERmodel_new(rank,0.5,0.5,data_train,label_train,M_dict[str(i)])

            P_t = RMmodel(rank,data_test)
            yt = P_t.dot(alpha)
            yt1 = np.argmax(yt,axis=1)

            pred_true = np.equal(label_test,yt1)
            acc = np.count_nonzero(pred_true) / len(pred_true)

            res_ser = pd.Series([split,i,rank,acc],index=['Split','M','R','Acc'])
            acc_MR = acc_MR.append(res_ser,ignore_index=True)
                #print(i,j, ":" ,acc)

    print(split, end="|")
    split += 1

acc_MR[['M','R']] = acc_MR[['M','R']].astype(int)
res = acc_MR.groupby(['M','R']).mean()
plot_df = res.unstack('M').loc[:,'Acc']

plot_df.plot(subplots=True,layout=[5,2],figsize=[10,10])
#plt.ylim(0.9,1)
plt.tight_layout()