###################################
### Made By: HERO Kwon          ###
### Date : 20181128             ###
### Desc : SPR Class HW#2       ###
###################################


import pandas as pd
import numpy as np
import os
import time
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import StratifiedKFold

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

#import data
## path
winedata_path = 'D:\\Matlab_Drive\\Data\\SPR\\UCI\\Wine\\'
optdata_path = 'D:\\Matlab_Drive\\Data\\SPR\\UCI\\Optdigit\\'
hw1data_path = 'D:\\Matlab_Drive\\Data\\SPR\\HW1\\'
## read: wine data
winedata = pd.read_csv(winedata_path+'wine.data',header=None)
winedata.columns = ['class','Alcohol','Malic acid','Ash','Alcalinity of ash','Magnesium',
                    'Total phenols','Flavanoids','Nonflavanoid phenols',
                    'Proanthocyanins','Color intensity','Hue',
                    'OD280/OD315 of diluted wines','Proline']
winedata['class'] = winedata['class'] - 1
## read: opt data
optdata_tra = pd.read_csv(optdata_path+'optdigits.tra',header=None)
optdata_tes = pd.read_csv(optdata_path+'optdigits.tes',header=None)
optdata = pd.concat([optdata_tra,optdata_tes],ignore_index=True)
optdata.rename(columns = {64:'class'},inplace=True)
## read: hw1 data
hw1data_train = pd.read_table(hw1data_path+'train.txt',delim_whitespace=True,names=['d1','d2','class'])
hw1data_test= pd.read_table(hw1data_path+'test.txt',delim_whitespace=True,names=['d1','d2','class'])
hw1data = pd.concat([hw1data_train,hw1data_test],ignore_index=True)
hw1data['class'] = hw1data['class'].values.astype('int')

# classification
data_name = {0:'hw1',1:'wine',2:'optdigit'}
data_3 = [hw1data,winedata,optdata]
df_acc = pd.DataFrame()
for i,data in enumerate(data_3):
    # K-Fold training data
    skf = StratifiedKFold(n_splits=5)
    split = 0
    data_v = data.drop('class',axis=1).values
    data_l = data['class'].values
    # iterate Split data
    for train_index,test_index in skf.split(data_v,data_l):
        data_train,data_test = data_v[train_index],data_v[test_index]
        label_train,label_test = data_l[train_index],data_l[test_index]
        # iterate Rank 1~5    
        for j in range(5):
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
            res_ser = pd.Series([data_name[i],'LSE',split,0,acc,time_elapsed],index=['Data','Model','Split','Rank','Acc','Time'])
            df_acc = df_acc.append(res_ser,ignore_index=True)
            #Test: RM
            time_now = time.time()
            Pt_rm = RMmodel(rank,data_test)
            yt_rm = Pt_rm.dot(alpha_rm)
            yt1_rm = np.argmax(yt_rm,axis=1)
            pred_true = np.equal(label_test,yt1_rm)
            acc = np.count_nonzero(pred_true) / len(pred_true)
            time_elapsed = time.time() - time_now
            res_ser = pd.Series([data_name[i],'RM',split,rank,acc,time_elapsed],index=['Data','Model','Split','Rank','Acc','Time'])
            df_acc = df_acc.append(res_ser,ignore_index=True)        
            #Test: TER
            time_now = time.time()
            Pt_ter = RMmodel(rank,data_test)
            yt_ter = Pt_ter.dot(alpha_ter)
            yt1_ter = np.argmax(yt_ter,axis=1)
            pred_true = np.equal(label_test,yt1_ter)
            acc = np.count_nonzero(pred_true) / len(pred_true)
            time_elapsed = time.time() - time_now
            res_ser = pd.Series([data_name[i],'TER',split,rank,acc,time_elapsed],index=['Data','Model','Split','Rank','Acc','Time'])
            df_acc = df_acc.append(res_ser,ignore_index=True)
        print('Data:' + data_name[i] + '/ Split:'+ str(split))
        split += 1