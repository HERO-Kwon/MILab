# MIT image : Analysis
# Made by : HERO Kwon
# Date : 190113

import os
import numpy as np
import pandas as pd
import pickle
import matplotlib.pyplot as plt
#import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import StratifiedKFold
import time

# data path
path_meta = '/home/herokwon/mount_data/Data/Wi-Fi_meta/'
path_csi = '/home/herokwon/mount_data/Data/Wi-Fi_processed/'
path_csi_np = '/home/herokwon/mount_data/Data/Wi-Fi_processed_npy/'
path_mit_image = '/home/herokwon/mount_data/Data/Wi-Fi_mit_image/'

#path_csi = 'D:\\Data\\Wi-Fi_processed\\'
#path_csi_np = 'D:\\Data\\Wi-Fi_processed_npy\\'
#path_meta = 'D:\\Data\\Wi-Fi_meta\\'
#path_sc = 'D:\\Data\\Wi-Fi_info\\'
#path_mit_image = 'D:\\Data\\Wi-Fi_mit_image\\'

# data info
df_info = pd.read_csv(path_meta+'data_subc_sig_v1.csv')
df_info = df_info[df_info.id_location==1]

person_uid = np.unique(df_info['id_person'])
dict_id = dict(zip(person_uid,np.arange(len(person_uid))))
csi_time = 50 #15000 #int(np.max(df_info['len']))
# parameters
max_value = np.max(df_info['max'].values)
#no_classes = len(np.unique(df_info['id_person']))
no_classes = len(dict_id)
csi_subc = 30
input_shape = (csi_time, csi_subc, 6)

# freq BW list
bw_list = pd.read_csv(path_meta+'wifi_f_bw_list.csv')

# 3D scan param
m,n = 2,3
c =  299792458 # speed of light 
#r = (160 + 160 + 164) * 0.01 # meter
r = 1.64 #meter
d = 45 * 0.01 # meter
ch = 8#3
max_subc = 30

# Load data
label_raw = np.load(path_csi_np + 'arr_lab.npy').astype('int')
arr_raw = np.load(path_csi_np + 'arr_abs.npy')
label_raw

#saved_images = [f.replace(".npy","") for f in os.listdir(path_mit_image)]
saved_images = ['S'+format(arr[0],'03d')+'_'+str(arr[1])+'_'+str(arr[2])+'_'+str(arr[3]) for arr in list(label_raw)]

df11 = df_info[(df_info.id_direction==1) & (df_info.id_location==1)]
df_target = df11[df11['id'].isin(saved_images)]
df_lab = df_target[['id','id_person','id_location','id_direction','id_exp']]
df_lab = df_lab.drop_duplicates()
df_lab
'''
data_mit = []
filename_mit = []
for file in df_lab.id.values:
    data_load = np.load(path_mit_image + file + '.npy')
    data_mit.append(data_load)
    filename_mit.append(file)
arr_mit = np.array(data_mit)
sum_mit = np.sum(arr_mit,axis=1).reshape([-1,20*20])
norm_mit = np.array([sum_mit[a,:] / np.max(sum_mit[a]) for a in range(arr_mit.shape[0])])
'''
label_mit = df_lab.id_person.values

arr_raw1 = arr_raw.reshape(891,15000,180)
diff_raw = np.diff(arr_raw1,axis=1)

s_idx = (np.arange(csi_time) * diff_raw.shape[0] / csi_time).astype('int')
s_raw = diff_raw[:,s_idx,:]


sum_raw = np.sum(s_raw,axis=1).reshape([-1,180])
norm_raw = np.array([sum_raw[a,:] / np.max(sum_raw[a]) for a in range(s_raw.shape[0])])


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

# change label
uniq_label = np.unique(label_mit)
label_table = pd.Series(np.arange(len(uniq_label)),index=uniq_label)
label_num = np.array([label_table[num] for num in label_mit])

# K-Fold training data
skf = StratifiedKFold(n_splits=5)
split = 0
data_v = norm_raw
data_l = label_num
for train_index,test_index in skf.split(data_v,data_l):
    data_train,data_test = data_v[train_index],data_v[test_index]
    label_train,label_test = data_l[train_index],data_l[test_index]


df_acc = pd.DataFrame()

# iterate Split data
for train_index,test_index in skf.split(data_v,data_l):
    data_train,data_test = data_v[train_index],data_v[test_index]
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
    
