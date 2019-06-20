import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedKFold
import time
from sklearn.metrics import confusion_matrix
import seaborn as sns
from sklearn.utils import shuffle
from scipy.optimize import brentq
from scipy.interpolate import interp1d
from sklearn.metrics import roc_curve


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
    dirs = [1,2,3,4]

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

def eer_graphs(dist_truth,dist_score,pos_label):
    fpr, tpr, thresholds = roc_curve(dist_truth, dist_score,pos_label=pos_label)
    eer = brentq(lambda x : 1. - x - interp1d(fpr, tpr)(x), 0., 1.)

    # ROC
    plt.plot(fpr, tpr, '.-')#, label=self.model_name)
    plt.plot([0, 1], [0, 1], 'k--', label="random guess")

    plt.xlabel('False Positive Rate (Fall-Out)')
    plt.ylabel('True Positive Rate (Recall)')
    plt.title('Receiver Operating Characteristic curve')
    plt.legend()
    plt.show()

    # EER
    #thres_mask = thresholds <= 5*np.median(dist_score)

    th_mask = thresholds#[thres_mask]
    fpr_mask= fpr#[thres_mask]
    tpr_mask = tpr#[thres_mask]

    plt.plot(th_mask,fpr_mask,color='blue', label="FPR")
    plt.plot(th_mask,1-tpr_mask,color='red',label="FNR")
    plt.plot(th_mask,fpr_mask + (1-tpr_mask),color='green',label="TER")
    plt.axhline(eer,color='black')
    plt.text(max(th_mask)*1.05,eer,'EER : ' + str(round(eer*100,1)))

    plt.xlabel('Thresholds')
    plt.ylabel('Error Rates (%)')
    plt.title('Equal Error Rate')
    plt.legend()
    plt.show()

    return(eer)
    
def make_val_triplets(x,y):
    n_samples = x.shape[0]
    xa = []
    xp = []
    xn = []
    for i in np.arange(0,n_samples-1,1):
        p_idxs = []
        for j in np.arange(i+1,n_samples,1):
            if(y[i]==y[j]):
                xa.append(x[i])
                xp.append(x[j])
                p_idxs.append(j)
        n_idxs = list(set(np.arange(0,n_samples)) - set(p_idxs))
        n_rsamp = np.random.choice(n_idxs,size=len(p_idxs),replace=False)
        xn.append(x[n_rsamp])
    
    arr_xa = np.array(xa).squeeze()
    arr_xp = np.array(xp).squeeze()
    arr_xn = np.vstack(xn).squeeze()
    
    return([arr_xa,arr_xp,arr_xn])