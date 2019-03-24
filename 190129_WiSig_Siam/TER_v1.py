import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedKFold
import time
import random
from sklearn.metrics import confusion_matrix
from keras.layers import Input, Conv2D, Lambda, merge, Dense, Flatten,MaxPooling2D
from keras.models import Model, Sequential
from keras.regularizers import l2
from keras import backend as K
from keras.optimizers import SGD,Adam
from keras.losses import binary_crossentropy
import numpy.random as rng
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.utils import shuffle
import keras

%matplotlib inline

# function to return key for any value 
def get_key(val,my_dict): 
    for key, list_val in my_dict.items(): 
         if val in list_val: 
             return key 
  
    return "key doesn't exist"


def ReadData(PATH,n_splits):
    print("loading data from {}".format(PATH))
    list_csi = []
    list_lab = []

    locs = [1]
    dirs = [1]

    # Filter Loc,Dir
    file_list = []
    for i in locs:
        for j in dirs:
            filename = 'Dataset_' + str(i) + '_' + str(j) + '.npy'
            file_list.append(filename)
    # Filter Dir
    
    for file in file_list:
        data_read = np.load(PATH + file)
        csi_read = data_read[:,4:].astype('float32')
        lab_read = data_read[:,0].astype('int')

        data_x = csi_read.reshape([-1,10,6,30,500]).swapaxes(2,4)

        uniq_label = np.unique(lab_read)
        label_table = pd.Series(np.arange(len(uniq_label)),index=uniq_label)
        data_y = np.array([label_table[num] for num in lab_read]).reshape([-1,10])
        
        # use half of the dataset
        idx_half = (data_y[:,0] < 50)
        
        list_csi.append(data_x[idx_half])
        list_lab.append(data_y[idx_half])
        
    arr_csi = np.array(list_csi).swapaxes(0,1).reshape([50*len(file_list),10,500,30,6])
    arr_lab = np.array(list_lab).swapaxes(0,1).reshape([50*len(file_list),10])

    skf = StratifiedKFold(n_splits,random_state=10)
    data_list = []

    arr_csi1 = arr_csi.reshape([-1,500,30,6])
    arr_lab1 = arr_lab.reshape([-1,1])

    for train_index,test_index in skf.split(arr_csi1,arr_lab1):
        
    #idx_tr,idx_te = train_test_split(np.arange(len(arr_csi)),test_size=0.2,random_state=10)
    #idx_tr = np.arange(len(arr_csi))[:600]
    #idx_te = np.arange(len(arr_csi))[600:]
        lab_ser = pd.Series(arr_lab1[:,0].astype('int'))
        lab_ser_tr = lab_ser.loc[train_index].reset_index()
        lab_ser_te = lab_ser.loc[test_index].reset_index()

        X = arr_csi1[train_index].reshape([-1,1,500,30,6])
        Xval = arr_csi1[test_index].reshape([-1,1,500,30,6])
        c = {}
        cval = {}
        Y = arr_lab1[train_index]
        Yval = arr_lab1[test_index]

        for num in np.unique(lab_ser_tr[0]):
            c[num] = list(lab_ser_tr[lab_ser_tr[0]==num].index)
        for num in np.unique(lab_ser_te[0]):
            cval[num] = list(lab_ser_te[lab_ser_te[0]==num].index)
            
        data_list.append([X,c,Xval,cval,Y,Yval])
    
    return(data_list)

def TERmodel(r,n,X,Y):
    alpha_ter = []
    for k in list(set(Y)):
        P_n = X[Y!=k]
        P_p = X[Y==k]
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
    return(np.array(alpha_ter).T)

def TERmodel_new(r,n,X,Y,mod_w):
    #simplified version of TER
    alpha = []
    for k in list(set(Y)):
        P = X

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


## Data
PATH = "/home/herokwon/mount_data/Data/Wi-Fi_HC/180_100/"

#Read Data
n_splits = 2
data_skf = ReadData(PATH,n_splits)
res_eer = pd.DataFrame()
for splits in range(2): #range(len(data_skf)):
    X,c,Xval,cval,Y,Yval = data_skf[splits]

    target_xs = np.squeeze(X)
    data_xs = np.mean(target_xs,axis=2).reshape([-1,500*6])
    target_xsv = np.squeeze(Xval)
    data_xsv = np.mean(target_xsv,axis=2).reshape([-1,500*6])
    data_y = np.squeeze(Y)#.reshape([-1,1])
    data_yv = np.squeeze(Yval)#.reshape([-1,1])

    for ii in range(1):
        t1 =time.time()
        alpha = TERmodel_new(0.5,0.5,data_xs,data_y,M_dict[str(ii)])
        data_test = data_xsv.dot(alpha)

        n_classes = data_test.shape[0]
        res_list = []
        
        for i in range(n_classes):
            for ti in range(n_classes):
                eer_dist = data_test[i,:] - data_test[ti,:]
                score = np.sqrt(np.sum(np.multiply(eer_dist,eer_dist)))

                score_list = [ii,splits,i,ti,int(data_yv[i] == data_yv[ti]),score]
                res_list.append(score_list)

        res_df = pd.DataFrame(res_list,columns=['ter_num','splits','img1','img2','TF','score'])
        res_df1 = res_df[res_df.img1 != res_df.img2]

        res_eer = pd.concat([res_eer,res_df1])

        print(str(ii) + '_EER Time:' + str(time.time() - t1))

def EER_Curve(tf_list,err_values,arr_thres):

    print("EER Curve")

    eer_df = pd.DataFrame(columns = ['thres','fn','fp','tn','tp'])
    

    for i, thres in enumerate(set(arr_thres)):
        predicted_tf = [e <= thres for e in err_values]
        
        tn, fp, fn, tp = confusion_matrix(tf_list,predicted_tf).ravel()

        eer_ser = {'thres':thres,'tn':tn,'fp':fp,'fn':fn,'tp':tp}
        eer_df = eer_df.append(eer_ser,ignore_index=True)
        
        curr_percent = 100 * (i+1) / len(arr_thres)
        if (curr_percent % 10)==0 : print(int(curr_percent),end="|")

    eer_df_graph = eer_df.sort_values(['thres'])
    eer_df_graph.fn = np.nan_to_num(eer_df_graph.fn / max(eer_df_graph.fn)) * 100
    eer_df_graph.fp = np.nan_to_num(eer_df_graph.fp / max(eer_df_graph.fp)) * 100
    eer_df_graph.te = eer_df_graph.fn + eer_df_graph.fp

    min_te_pnt = eer_df_graph[eer_df_graph.te == min(eer_df_graph.te)]
    min_te_val = float((np.unique(min_te_pnt['fn'].values) + np.unique(min_te_pnt['fp'].values)) / 2)

    plt.plot(eer_df_graph.thres,eer_df_graph.fn,color='red',label='FNR')
    plt.plot(eer_df_graph.thres,eer_df_graph.fp,color='blue',label='FPR')
    plt.plot(eer_df_graph.thres,eer_df_graph.te,color='green',label='TER')
    plt.axhline(min_te_val,color='black')
    plt.text(max(eer_df_graph.thres)*0.9,min_te_val-10,'EER : ' + str(round(min_te_val,2)))
    plt.legend()
    plt.title("EER Curve")

    plt.show()

    return(min_te_val,eer_df)

list_eer = []
for s in range(2): #range(2):
    for ii in range(1):
        res_plot = pd.DataFrame()
        res_plot = res_eer[(res_eer.splits==s)&(res_eer.ter_num==ii)]
        print(s,ii)
        res_eer_val,_ = EER_Curve(res_plot.TF,res_plot.score,np.arange(0,5,0.01))
        list_eer.append(res_eer_val)
        