import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
from karnet_v1 import KARnet
from scipy.optimize import brentq
from scipy.interpolate import interp1d
from sklearn.metrics import roc_curve
from sklearn.model_selection import StratifiedKFold
from keras.utils import to_categorical

## Functions
# function to return key for any value 
def get_key(val,my_dict): 
    for key, list_val in my_dict.items(): 
         if val in list_val: 
             return key 
  
    return "key doesn't exist"

def ReadCSIData(PATH,ratio,locs,dirs):
    print("loading data from {}".format(PATH))
    list_csi = []
    list_lab = []
    list_res_csi = []
    list_res_lab = []
    
    ids = int(ratio * 100)
    res_ids = int((1-ratio)*100)

    # Filter Loc,Dir
    file_list = []
    for i in locs:
        for j in dirs:
            filename = 'Dataset_' + str(i) + '_' + str(j) + '.npy'
            file_list.append(filename)

    for file in file_list:
        data_read = np.load(PATH + file)
        csi_read = data_read[:,4:].astype('float32')
        lab_read = data_read[:,0].astype('int')

        data_x = csi_read.reshape([-1,10,6,30,500]).swapaxes(2,4)

        uniq_label = np.unique(lab_read)
        label_table = pd.Series(np.arange(len(uniq_label)),index=uniq_label)
        data_y = np.array([label_table[num] for num in lab_read]).reshape([-1,10])
        
        # use half of the dataset
        idx_ids = (data_y[:,0] < ids)
        list_csi.append(data_x[idx_ids])
        list_lab.append(data_y[idx_ids])
        
        reserve_ids = (data_y[:,0] >= ids)
        list_res_csi.append(data_x[reserve_ids])
        list_res_lab.append(data_y[reserve_ids])

    arr_csi = np.array(list_csi).swapaxes(0,1).reshape([ids*len(file_list),10,500,30,6])
    arr_lab = np.array(list_lab).swapaxes(0,1).reshape([ids*len(file_list),10])
    Xs = arr_csi.reshape([-1,1,500,30,6])
    Ys = arr_lab.reshape([-1,1])

    reserve_csi = np.array(list_res_csi).swapaxes(0,1).reshape([res_ids*len(file_list),10,500,30,6])
    reserve_lab = np.array(list_res_lab).swapaxes(0,1).reshape([res_ids*len(file_list),10])
    reserve_Xs = reserve_csi.reshape([-1,1,500,30,6])
    reserve_Ys = reserve_lab.reshape([-1,1])

    return([Xs,Ys,reserve_Xs,reserve_Ys])


## Data
#PATH = "/home/herokwon/mount_data/Data/Wi-Fi_HC/180_100/"
PATH = "D:\\Data\\WIFI\\Wi-Fi_HC\\180_100\\"
Xs,Ys,resXs,resYs = ReadCSIData(PATH,use_ratio,locs,dirs)
prep_xs = np.squeeze(Xs).reshape([-1,500*30*6])#np.mean(np.squeeze(Xs),axis=2).reshape([-1,500*6])
#prep_xs = np.squeeze(Xs)[:,:,15,:].reshape([-1,500*6])
prep_ys = np.squeeze(Ys)

#m_results = pd.DataFrame(columns = ['name','splits','splits_number','random_state','eer','param0','tr_time'])

skf = StratifiedKFold(n_splits,random_state=10)

h_list = []
#h_list.append([1048576,65536,1024])
#h_list.append([1048576,1024])

h_list.append([2048,1024])
#h_list.append([2048,2048,1024])
#h_list.append([4096,1024])
#h_list.append([4096,2048,1024])
#h_list.append([1024])
#h_list.append([2048])
#h_list.append([4096])
#h_list.append([8192,1024])
#h_list.append([16384,1024])
#h_list.append([512,1024])
#h_list.append([256,1024])

import time

for idx_tr,idx_te in skf.split(prep_xs,prep_ys):
    for h in h_list:
        print(h)
        t1 = time.time()
        x = prep_xs[idx_tr]
        y = to_categorical(prep_ys[idx_tr])
        xte = prep_xs[idx_te]
        yte = to_categorical(prep_ys[idx_te])        

        data_list = [x,y,xte,yte]
        try:
            kar = KARnet(data_list,h=h,rseed=8)

            kar.train()

            acc_tr = kar.accuracy(mode='train')
            print(acc_tr)
            acc_te = kar.accuracy(mode='test')
            print(acc_te)

            print(time.time()-t1)
        except:
            pass