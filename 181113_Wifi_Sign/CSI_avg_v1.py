# Getting CSI Avg
# Made by : HERO Kwon
# Date : 190108

import os
import numpy as np
import pandas as pd
import pickle
import gzip
import matplotlib.pyplot as plt

# data path_GCP
#path_meta = '/home/herokwon/mount_data/Data/Wi-Fi_meta/'
#path_csi = '/home/herokwon/mount_data/Data/Wi-Fi_processed/'
#path_csi_np = '/home/herokwon/mount_data/Data/Wi-Fi_processed_npy/'

# data path_mi
path_csi = 'D:\\Data\\Wi-Fi_processed\\'
path_csi_np = 'D:\\Data\\Wi-Fi_processed_npy\\'
path_meta = 'D:\\Data\\Wi-Fi_meta\\'


# data info
df_info = pd.read_csv('data_subc_sig_v1.csv')
person_uid = np.unique(df_info['id_person'])
dict_id = dict(zip(person_uid,np.arange(len(person_uid))))
csi_time = 15000 #int(np.max(df_info['len']))

# data generator
def gen_csi(df_info,len_num):
    for file in np.unique(df_info.id.values):
        # read sample data
        # load and uncompress.
        with gzip.open(path_csi+file+'.pickle.gz','rb') as f:
            data1 = pickle.load(f)
        data1_diff = data1 #np.diff(data1,axis=0)
        # zero pad
        pad_len = len_num - data1_diff.shape[0]
        data1_pad = np.pad(data1_diff,((0,pad_len),(0,0),(0,0),(0,0)),'constant',constant_values=0)

        yield(data1_pad)

# make avgcsi dict
dict_avgcsi = {}
for num,person_id in enumerate(person_uid):
    df1 = df_info[df_info.id_person == person_id]
    df1_uid = np.unique(df1.id.values)

    sum_array = np.zeros([csi_time,30,2,3],dtype=complex)
    num_array = 0
    gen = gen_csi(df1,csi_time)

    for i in range(len(df1_uid)):
        try:
            array1 = next(gen)
            num_array += 1
        except:
            pass
        sum_array += array1
    
    avg_array = sum_array / num_array

    # phase diff
    ph_array = np.angle(avg_array)
    dph_array = np.diff(ph_array,axis=0)
    dph_mean = np.mean(dph_array,axis=(0,2,3))

    # SAVE DICT
    dict_avgcsi[person_id] = (avg_array, dph_mean)

    print('finished: ' + str(person_id))
    print(str(num) +'/'+ str(len(person_uid)))

with open(path_meta + 'dict_avgcsi.pickle','wb') as f:
    pickle.dump(dict_avgcsi,f,pickle.HIGHEST_PROTOCOL)


