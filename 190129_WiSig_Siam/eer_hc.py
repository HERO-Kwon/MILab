import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedKFold
import time
import random
from sklearn.metrics import confusion_matrix

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

        X = arr_csi1[train_index]
        Xval = arr_csi1[test_index]
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
    
## Data
PATH = "/home/herokwon/mount_data/Data/Wi-Fi_HC/180_100/"

#Read Data
n_splits = 2
data_skf = ReadData(PATH,n_splits)

def Fe_HC(target_avg,c,s,num_eig):
    X = target_avg.T
    ones = np.ones(X.shape[1]).reshape([-1,1])
    u = np.dot(1/(c*s) * X,ones)
    A = X - np.dot(u,ones.T)
    Z = np.dot(A,A.T)
    V,W = np.linalg.eig(Z)
    return(W[:num_eig,:],u)

res_eer = pd.DataFrame()
for splits in range(len(data_skf)):
    X,c,Xval,cval,Y,Yval = data_skf[splits]

    c = len(np.unique(Y))
    s = int(Y.shape[0] / c)

    target_xs = X#.reshape([-1,500,30,6])
    data_xs = np.mean(target_xs,axis=2).reshape([-1,500*6])
    target_xsv = Xval.reshape([-1,500,30,6])
    data_xsv = np.mean(target_xsv,axis=2).reshape([-1,500*6])
    data_y = Y#.reshape([-1,1])
    data_yv = Yval.reshape([-1,1])

    W,u = Fe_HC(data_xs,c,s,40)
    data_train = np.dot(W,data_xs.T-u).T
    data_test = np.dot(W,data_xsv.T-u).T


    n_classes = data_test.shape[0]

    res_list = []
    t1 =time.time()
    for i in range(n_classes):
        for ti in range(n_classes):
            eer_dist = data_test[i,:] - data_test[ti,:]
            score = np.abs(np.sqrt(np.sum(np.multiply(eer_dist,eer_dist))))

            score_list = [splits,i,ti,int(data_yv[i][0] == data_yv[ti][0]),score]
            res_list.append(score_list)

    res_df = pd.DataFrame(res_list,columns=['splits','img1','img2','TF','score'])
    res_df.head()
    res_df1 = res_df[res_df.img1 != res_df.img2]

    res_eer = pd.concat([res_eer,res_df1])

    print('EER Time:' + str(time.time() - t1))


## Function : EER Curve

## Function : EER Curve

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
    eer_df_graph.fn = eer_df_graph.fn / max(eer_df_graph.fn) * 100
    eer_df_graph.fp = eer_df_graph.fp / max(eer_df_graph.fp) * 100
    eer_df_graph.te = eer_df_graph.fn + eer_df_graph.fp

    min_te_pnt = eer_df_graph[eer_df_graph.te == min(eer_df_graph.te)]
    min_te_val = float((min_te_pnt['fn'].values + min_te_pnt['fp'].values) / 2)

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
for s in range(len(data_skf)):
    res_plot = res_eer[res_eer.splits==s]
    res_eer_val,_ = EER_Curve(res_plot.TF,res_plot.score,np.arange(1,100))
    list_eer.append(res_eer_val)

np.mean(list_eer)