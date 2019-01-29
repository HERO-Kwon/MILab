# MIT image
# Made by : HERO Kwon
# Date : 190108

import os
import numpy as np
import pandas as pd
import pickle
import gzip
import matplotlib.pyplot as plt
import math
import cmath
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import StratifiedKFold


def Fe_HC(target_avg,c,s,num_eig):
    X = target_avg.T
    ones = np.ones(X.shape[1]).reshape([-1,1])
    u = np.dot(1/(c*s) * X,ones)
    A = X - np.dot(u,ones.T)
    Z = np.dot(A,A.T)
    V,W = np.linalg.eig(Z)
    return(W[:num_eig,:],u)

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

#Models : LSE,RM,TER
def Model3(rank,r,n,X,Y):
    # LSE
    alpha_lse = []
    for k in list(set(Y)):
        P_lse = X
        I_lse = np.eye(P_lse.shape[1])
        b_lse = 10**(-4)
        y_lse = (Y==k).astype('int')
        ak_lse = np.dot(np.dot(np.linalg.pinv(b_lse*I_lse + P_lse.T.dot(P_lse)),P_lse.T),y_lse)
        alpha_lse.append(ak_lse)    
    # RM
    alpha_rm = []
    for k in list(set(Y)):
        P_rm = RMmodel(rank,X)
        I_rm = np.eye(P_rm.shape[1])
        b_rm = 10**(-4)
        y_rm = (Y==k).astype('int')
        ak_rm = np.dot(np.dot(np.linalg.pinv(b_rm*I_rm + P_rm.T.dot(P_rm)),P_rm.T),y_rm)
        alpha_rm.append(ak_rm)  
    # TER
    alpha_ter = []
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
        alpha_ter.append(ak)
    return(np.array(alpha_lse).T, np.array(alpha_rm).T,np.array(alpha_ter).T)


# data path_mi
path_csi = 'D:\\Data\\Wi-Fi_processed\\'
path_csi_np = 'D:\\Data\\Wi-Fi_processed_npy\\'
path_csi_hc = 'D:\\Data\\Wi-Fi_HC\\180_100\\'
path_csi_hc_dco = 'D:\\Data\\Wi-Fi_HC\\180_100_DCout\\'
path_meta = 'D:\\Data\\Wi-Fi_meta\\'
path_sc = 'D:\\Data\\Wi-Fi_info\\'
path_mit_image = 'D:\\Data\\Wi-Fi_mit_image\\'
path_csism = 'D:\\Data\\Wi-Fi_sm_npy\\'

# load data, label
path_file = path_csi_hc
list_read = []
for filename in os.listdir(path_file):
    list_read.append(np.load(path_file + filename))
    print(filename)

arr_read = np.array(list_read)
target_data = arr_read[0]
target_xs = target_data[:,4:].astype('float32').reshape([-1,500,30,6])
data_xs = np.mean(target_xs,axis=2).reshape([-1,500*6])

# make label
label_data = target_data[:,0].astype('int')
uniq_label = np.unique(label_data)
label_table = pd.Series(np.arange(len(uniq_label)),index=uniq_label)
data_y = np.array([label_table[num] for num in label_data])

c = len(np.unique(label_data))
s = 10

from skimage.transform import resize
# K-Fold training data
skf = StratifiedKFold(n_splits=5)
split = 0
data_v = data_xs
data_l = data_y

df_acc = pd.DataFrame()
import time
# iterate Split data
for train_index,test_index in skf.split(data_v,data_l):
    #data_train,data_test = data_v[train_index],data_v[test_index]
    W,u = Fe_HC(data_v[train_index],c,s,40)
    data_train = np.dot(W,data_v[train_index].T-u).T
    data_test = np.dot(W,data_v[test_index].T-u).T
    label_train,label_test = data_l[train_index],data_l[test_index]
    # iterate Rank 1~5    
    for j in range(1):
        rank = j+1
        #Traning Result of models
        alpha_lse, alpha_rm, alpha_ter = Model3(rank,0.5,0.5,data_train,label_train)
        #Test: LSE
        time_now = time.time()
        Pt_lse = data_test
        yt_lse = Pt_lse.dot(alpha_lse)
        yt1_lse = np.argmax(yt_lse,axis=1)
        pred_true = np.equal(label_test,yt1_lse)
        acc = np.count_nonzero(pred_true) / len(pred_true)
        time_elapsed = time.time() - time_now
        res_ser = pd.Series([data_train,'LSE',split,0,acc,time_elapsed],index=['Data','Model','Split','Rank','Acc','Time'])
        df_acc = df_acc.append(res_ser,ignore_index=True)
        #Test: RM
        time_now = time.time()
        Pt_rm = RMmodel(rank,data_test)
        yt_rm = Pt_rm.dot(alpha_rm)
        yt1_rm = np.argmax(yt_rm,axis=1)
        pred_true = np.equal(label_test,yt1_rm)
        acc = np.count_nonzero(pred_true) / len(pred_true)
        time_elapsed = time.time() - time_now
        res_ser = pd.Series([data_train,'RM',split,rank,acc,time_elapsed],index=['Data','Model','Split','Rank','Acc','Time'])
        df_acc = df_acc.append(res_ser,ignore_index=True)        
        #Test: TER
        time_now = time.time()
        Pt_ter = RMmodel(rank,data_test)
        yt_ter = Pt_ter.dot(alpha_ter)
        yt1_ter = np.argmax(yt_ter,axis=1)
        pred_true = np.equal(label_test,yt1_ter)
        acc = np.count_nonzero(pred_true) / len(pred_true)
        time_elapsed = time.time() - time_now
        res_ser = pd.Series([data_train,'TER',split,rank,acc,time_elapsed],index=['Data','Model','Split','Rank','Acc','Time'])
        df_acc = df_acc.append(res_ser,ignore_index=True)
    print('Data:' + '/ Split:'+ str(split))
    split += 1

df_acc.groupby('Model').mean()