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

# data path
#path_meta = '/home/herokwon/mount_data/Data/Wi-Fi_meta/'
#path_csi = '/home/herokwon/mount_data/Data/Wi-Fi_processed/'
#path_csi_np = '/home/herokwon/mount_data/Data/Wi-Fi_processed_npy/'

# data path_mi
path_csi = 'D:\\Data\\Wi-Fi_processed\\'
path_csi_np = 'D:\\Data\\Wi-Fi_processed_npy\\'
path_meta = 'D:\\Data\\Wi-Fi_meta\\'
path_sc = 'D:\\Data\\Wi-Fi_info\\'

# data info
df_info = pd.read_csv(path_meta+'data_subc_sig_v1.csv')
person_uid = np.unique(df_info['id_person'])
dict_id = dict(zip(person_uid,np.arange(len(person_uid))))
csi_time = 15000 #int(np.max(df_info['len']))
# parameters
max_value = np.max(df_info['max'].values)
#no_classes = len(np.unique(df_info['id_person']))
no_classes = len(dict_id)
csi_subc = 30
input_shape = (csi_time, csi_subc, 6)

# freq BW list
bw_list = pd.read_csv(path_meta+'wifi_f_bw_list.csv')

# avg Array
with open(path_meta + 'dict_avgcsi.pickle','rb') as f:
    dict_avg = pickle.load(f)


# make data generator
def gen_csi(df_info,dict_avg,id_num,len_num):
    for file in np.unique(df_info.id.values):
        # read sample data
        # load and uncompress.
        with gzip.open(path_csi+file+'.pickle.gz','rb') as f:
            data1 = pickle.load(f)
        data1_diff = data1 #np.diff(data1,axis=0)
        # zero pad
        pad_len = len_num - data1_diff.shape[0]
        data1_pad = np.pad(data1_diff,((0,pad_len),(0,0),(0,0),(0,0)),'constant',constant_values=0)

        # subcarrier info
        data1_sc_df = pd.read_csv(path_sc + file + '_df_sc.csv')
        data1_time = data1_sc_df['timestamp_low']
        data1_time_pad = np.pad(data1_time,((0,pad_len)),'constant',constant_values=0)
        # Label
        id_key = df_info[df_info.id==file]['id_person'].values[0].astype('int')
        data1_y = dict_id[id_key]
        
        # subtract average
        arr_avg = dict_avg[id_key][0]
        data1_result = data1_pad - arr_avg
        
        yield(data1_result ,data1_y,id_key,data1_time_pad)

gen = gen_csi(df_info,dict_avg,no_classes,csi_time)
target_sig,target_lab,target_id,target_time = next(gen)

# 3D scan
m,n = 2,3
c =  299792458 # speed of light 
r = (160 + 160 + 164) * 0.01 # meter
#r = 1.64 #meter
d = 45 * 0.01 # meter
max_ch = 1#3
max_subc = 30

th_range,si_range = (30,30)
sig_mat = np.zeros([max_subc,csi_time,2*th_range,2*si_range])

# 3D Reconstruction func
from numba import vectorize
@vectorize(['complex128(complex128,float32,float32,int32,int32,\
float32,float32,float32,float32,int32,int32)'], target='cpu')
def Recon_3d(sig,theta,sigma,m,n,lam,d,k,r,t,c):
    above_eq1 = 1j * (2*math.pi) * k * r * t / c
    above_eq2 = 1j * (2*math.pi/lam) * math.sin(theta) * ((n+1)*d*math.cos(sigma) + (m+1)*d*math.sin(sigma))
    eq_res = sig* cmath.exp(above_eq1) * cmath.exp(above_eq2)
    return eq_res #math.atan2(eq_res.imag,eq_res.real)


# Calc 3D
# Calc 3D
for subc in range(max_subc):
    sig1 = target_sig[:,subc,:,:]
    for idx_th,i in enumerate(range(-th_range,th_range)):
        for idx_si,j in enumerate(range(-si_range,si_range)):
            lam =  c*0.000001 / bw_list[str(subc)][8] #wavelength = 300 / frequency in MHz
            t = np.arange(1,csi_time+1,1,dtype=np.int32)
            k = 0.0
            theta = i * (np.radians(360)/2) / th_range
            sigma = j * (np.radians(360)/2) / si_range
            sum_eq = np.zeros(csi_time,dtype=np.complex_)
            for m in [0,1]: 
                for n in [0,1,2]: 
                    #above_eq1 = Calc_Above1(k,r,t,c)
                    sig1 = np.ascontiguousarray(target_sig[:,subc,m,n], dtype=np.complex128)
                    sum_eq += Recon_3d(sig1,theta,sigma,m,n,lam,d,k,r,t,c)
            sig_mat[subc,:,idx_th,idx_si] =  np.angle(sum_eq)

import seaborn as sns
ax = sns.heatmap(sig_mat[0,0,:th_range,:si_range])#,vmin=np.median(sig_mat))

