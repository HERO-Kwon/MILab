import os
import tensorflow as tf
import numpy as np
import pandas as pd
import pickle
import gzip
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

# data path
path_csi =  'J:\\Data\\Wi-Fi_processed\\'
path_csi_hc = 'J:\\Data\\Wi-Fi_HC\\180_100\\'

# data info
df_info = pd.read_csv('data_subc_sig_v1.csv')
#df_info = df_info[df_info.id_person < 50]

person_uid = np.unique(df_info['id_person'])
csi_uid = np.unique(df_info['id'])
dict_id = dict(zip(person_uid,np.arange(len(person_uid))))

# parameters
max_value = np.max(df_info['max'].values)
#no_classes = len(np.unique(df_info['id_person']))
no_classes = len(dict_id)
csi_time = 15000 #int(np.max(df_info['len']))
csi_subc = 30
input_shape = (csi_time, csi_subc, 6)


import h5py
#Make HDF5
with h5py.File(path_csi+'csi.hdf5','w') as f:
    f.create_dataset('csi',(len(csi_uid),csi_time,csi_subc,2,3),dtype='complex128')
    f.create_dataset('label',(len(csi_uid),4),dtype='float32')
    csi_set = f['csi']
    label_set = f['label']
    
    for i,file in enumerate(csi_uid):
        with gzip.open(path_csi+file+'.pickle.gz','rb') as f:
            data1 = pickle.load(f)
            # zero pad
            pad_len = csi_time - data1.shape[0]
            data1_pad = np.pad(data1,((0,pad_len),(0,0),(0,0),(0,0)),'constant',constant_values=0)

            # Label
            data1_info = df_info[df_info.id==file].iloc[0]
            data1_label = np.array([data1_info.id_person,
                      data1_info.id_location,
                      data1_info.id_direction,
                      data1_info.id_exp])
        label_set[i] = data1_label.astype('float32')
        csi_set[i] = data1_pad