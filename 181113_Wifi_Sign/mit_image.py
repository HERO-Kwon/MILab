import os
import tensorflow as tf
import numpy as np
import pandas as pd
import pickle
import gzip
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import math
import cmath

# data path
path_csi =  'J:\\Data\\Wi-Fi_processed\\'
path_csi_hc = 'J:\\Data\\Wi-Fi_HC\\180_100\\'

# data info
df_info = pd.read_csv('data_subc_sig_v1.csv')
#df_info = df_info[df_info.id_person < 50]
df_info[(df_info.id_location==1)  & (df_info.id_direction==1)]

person_uid = np.unique(df_info['id_person'])
dict_id = dict(zip(person_uid,np.arange(len(person_uid))))

# parameters
max_value = np.max(df_info['max'].values)
#no_classes = len(np.unique(df_info['id_person']))
no_classes = len(dict_id)
csi_time = int(np.max(df_info['len']))
csi_subc = 30
input_shape = (csi_time, csi_subc, 6)

# make data generator
def gen_csi(df_info,id_num,len_num):
    for file in np.unique(df_info.id.values):
        # read sample data
        # load and uncompress.
        with gzip.open(path_csi+file+'.pickle.gz','rb') as f:
            data1 = pickle.load(f)
        data1_diff = data1
        # zero pad
        pad_len = len_num - data1_diff.shape[0]
        data1_pad = np.pad(data1_diff,((0,pad_len),(0,0),(0,0),(0,0)),'constant',constant_values=0)

        # Label
        id_key = df_info[df_info.id==file]['id_person'].values[0].astype('int')
        data1_y = dict_id[id_key]

        yield(data1_pad ,data1_y)

gen = gen_csi(df_info,no_classes,csi_time)
# 3D scan
m,n = 2,3
r = (160 + 160 + 164) * 0.01 # meter
d = 45 * 0.01 # meter
lam = 300 / 2450 #wavelength = 300 / frequency in MHz

theta,sigma = math.pi,math.pi
above_eq = 1j * (2*math.pi/lam) * math.sin(theta) * (n*d*math.cos(sigma) + m*d*math.sin(sigma))

def calc_p(target_sig,theta,sigma):
    sum_eq = np.complex(0)
    for i in range(m):
        for j in range(n):
            for k in range(csi_time):
                above_eq = 1j * (2*math.pi/lam) * math.sin(theta) * (n*d*math.cos(sigma) + m*d*math.sin(sigma))
                sum_eq += target_sig[k,i,j] * cmath.exp(above_eq)
    return(np.abs(sum_eq))

sig_mat = np.zeros([30,100,100])

for subc in range(30):
    for i in range(100):
        for j in range(100):
            theta = i * math.pi / 100 
            sigma = j * math.pi / 100
            sig_mat[subc,i,j] = calc_p(csi[:,subc,:,:],theta,sigma)
            print((subc,i,j))