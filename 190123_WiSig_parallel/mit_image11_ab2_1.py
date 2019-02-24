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
#path_csi = '/home/herokwon/mount_data/Data/Wi-Fi_processed_sm/'
#path_csi_np = '/home/herokwon/mount_data/Data/Wi-Fi_processed_npy/'
#path_mit_image = '/home/herokwon/mount_data/Data/Wi-Fi_re_deb1/'
#path_csi_hc = '/home/herokwon/mount_data/Data/Wi-Fi_HC/180_100/'
# data path_mi
path_csi = 'D:\\Data\\Wi-Fi_processed_sm\\'
path_csi_np = 'D:\\Data\\Wi-Fi_processed_npy\\'
path_meta = 'D:\\Data\\Wi-Fi_meta\\'
path_sc = 'D:\\Data\\Wi-Fi_info\\'
path_mit_image = 'G:\\Data\\Wi-Fi_11_mithc'
path_csi_hc = 'D:\\Data\\Wi-Fi_HC\\180_100\\'

data_num = 0
data_i = 1

# 3D Reconstruction func
from numba import vectorize
@vectorize(['complex64(complex64,float32,float32,int32,int32,float32,float32,float32,int32)'], target='cpu')
def Recon3d(sig,theta,sigma,m,n,lam,d,r,c):
    #above_eq1 = 1j * (2*math.pi) * k * r * t / c
    #above_eq1 = 1j * (2*math.pi) * c * dt * 0.000001 / lam
    above_eq1 = 1j * (2*math.pi) * r / lam
    above_eq2 = 1j * (2*math.pi/lam) * math.sin(theta) * ((n+1)*d*math.cos(sigma) + (m+1)*d*math.sin(sigma))
    eq_res = sig* cmath.exp(above_eq1) * cmath.exp(above_eq2)
    #eq_res = cmath.exp(above_eq1) * cmath.exp(above_eq2)
    return eq_res #math.atan2(eq_res.imag,eq_res.real)


def Calc3d(target_sig,max_subc,theta,sigma,bw_list,ch,d,r,c,m,n):
    #vectorize arrays
    subc_array = np.ones([csi_time,max_subc,m,n])
    m_array = np.ones([csi_time,max_subc,m,n])
    n_array = np.ones([csi_time,max_subc,m,n])
    for i in range(max_subc):
        subc_array[:,i,:,:] = i
    for i in range(m):
        m_array[:,:,i,:] = i
    for i in range(n):
        n_array[:,:,:,i] = i
    subc_array = subc_array.flatten().astype(np.int32)
    m_array = m_array.flatten().astype(np.int32)
    n_array = n_array.flatten().astype(np.int32)
    lam_array = np.array([c*0.000001/bw_list[str(e)][ch] for e in subc_array],dtype=np.float32)
    
    target_array = target_sig.flatten().astype(np.complex64)
    
    #3d recon
    sum_eq = np.zeros(csi_time*max_subc*m*n,dtype=np.complex_)
    sum_eq = Recon3d(target_array,theta,sigma,m_array,n_array,lam_array,d,r,c)

    sum_reshape = np.sum(sum_eq.reshape([-1,max_subc,m,n]),axis=(1,2,3))
    return(np.abs(sum_reshape))


# load data, label
path_file = path_csi_hc
list_read = []
for filename in [os.listdir(path_file)[data_num]]:
    list_read.append(np.load(path_file + filename))
    print(filename)

arr_read = np.array(list_read)[:,200*data_i:200*(data_i+1),:]
target_data = arr_read[0]
target_xs = target_data[:,4:].astype('float32').reshape([-1,2,3,30,500])
data_xs = np.mean(target_xs,axis=3).reshape([-1,500*6])

# make label
label_data = target_data[:,0].astype('int')
uniq_label = np.unique(label_data)
label_table = pd.Series(np.arange(len(uniq_label)),index=uniq_label)
data_y = np.array([label_table[num] for num in label_data])

target_string = target_data[:,:4].astype('int')
list_filename = ['_'.join([str(target_string[i,0]),str(target_string[i,1]),str(target_string[i,2]),str(target_string[i,3])]) for i in range(len(target_string))]

target_sig = target_xs.swapaxes(1,4).swapaxes(2,3).swapaxes(3,4)
#target_sig = target_xs.reshape([-1,500,30,2,3])

csi_time = 500
th_range,si_range = (5,5)
max_subc = 30
# freq BW list
bw_list = pd.read_csv(path_meta+'wifi_f_bw_list_v1.csv')
# 3D scan param
m,n = 2,3
c =  299792458 # speed of light 
#r = (160 + 160 + 164) * 0.01 # meter
r = 1.64 #meter
d = 45 * 0.01 # meter
ch = 7#3
max_subc = 30

infiles = set([file.replace(".npy","") for file in os.listdir(path_mit_image)])
tofiles = list(set(list_filename) - infiles)
for k,f in enumerate(tofiles):
    
    target_sig = target_xs[k]

    #target_sig,target_lab,target_id,target_file = next(gen)
    th_range,si_range = (5,5)
    sig_mat = np.zeros([csi_time,2*th_range,2*si_range])

    import time
    t1 = time.time()
    for idx_th,i in enumerate(range(-th_range,th_range)):
        for idx_si,j in enumerate(range(-si_range,si_range)):
            theta = i * (np.radians(60)/2) / th_range
            sigma = j * (np.radians(60)/2) / si_range
            sig_mat[:,idx_th,idx_si] = Calc3d(target_sig,max_subc,theta,sigma,bw_list,ch,d,r,c,m,n)
    print(time.time()-t1)
    print(f)

    np.save(path_mit_image + f + '.npy',sig_mat)