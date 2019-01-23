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

# 3D Reconstruction func
from numba import vectorize
from numba import jit
@vectorize(['complex64(complex64,float32,float32,int32,int32,float32,float32,float32,int32)'], target='cuda')
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
    subc_array = np.zeros([csi_time,max_subc,m,n])
    m_array = np.zeros([csi_time,max_subc,m,n])
    n_array = np.zeros([csi_time,max_subc,m,n])
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
    #lam = bw_list['F0 (MHz)'][ch]
    target_array = target_sig.flatten().astype(np.complex64)
    
    #3d recon
    sum_eq = np.zeros(csi_time*max_subc*m*n,dtype=np.complex_)
    sum_eq = Recon3d(target_array,theta,sigma,m_array,n_array,lam_array,d,r,c)
    sum_reshape = np.sum(sum_eq.reshape([-1,max_subc,m,n]),axis=(1,2,3))
    return(np.abs(sum_reshape))

import matplotlib.animation as animation
import numpy as np
from pylab import *

dpi = 100

def ani_frame(arr_mov,file):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.set_aspect('equal')
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    
    im = ax.imshow(arr_mov[0,:,:],cmap='gray',interpolation='nearest')
    #im.set_clim([0,1])
    fig.set_size_inches([3,3])
    
    tight_layout()

    def update_img(n):
        tmp = arr_mov[n,:,:]
        im.set_data(tmp)
        return im

    #legend(loc=0)
    ani = animation.FuncAnimation(fig,update_img,arr_mov.shape[0])#,interval=1)
    writer = animation.writers['ffmpeg'](fps=5)

    ani.save(path_movie + file + '.mp4',writer=writer,dpi=dpi)
    
    # save sum image
    fig1 = plt.figure()
    ax1 = fig1.add_subplot(111)
    ax1.set_aspect('equal')
    ax1.get_xaxis().set_visible(False)
    ax1.get_yaxis().set_visible(False)
    im1 = ax1.imshow(np.sum(arr_mov,axis=0),cmap='gray',interpolation='nearest')
    fig1.set_size_inches([3,3])
    tight_layout()
    fig1.savefig(path_movie + file+'_sum.png')
    
    plt.close('all')
    return ani


# data path
path_meta = '/home/herokwon/mount_data/Data/Wi-Fi_meta/'
path_csi = '/home/herokwon/mount_data/Data/Wi-Fi_processed_sm/'
path_csi_np = '/home/herokwon/mount_data/Data/Wi-Fi_processed_npy/'
path_mit_image = '/home/herokwon/mount_data/Data/Wi-Fi_mit_denoise_debug/'
path_movie = '/home/herokwon/mount_data/Data/Wi-Fi_movie/'

# data path_mi
#path_csi = 'J:\\Data\\Wi-Fi_processed\\'
#path_csi_np = 'J:\\Data\\Wi-Fi_processed_npy\\'
#path_meta = 'J:\\Data\\Wi-Fi_meta\\'
#path_sc = 'J:\\Data\\Wi-Fi_info\\'
#path_mit_image = 'J:\\Data\\Wi-Fi_mit_image\\'

# data info
df_info = pd.read_csv(path_meta+'data_subc_sig_v1.csv') # 55429 rows
#df_info = df_info[df_info.id_location==1]

#df_info = df_info.iloc[0:3700]
#df_info = df_info.iloc[3700:7400]
#df_info = df_info.iloc[7400:11100]
#df_info = df_info.iloc[11100:14800]
#df_info = df_info.iloc[14800:18500]
#df_info = df_info.iloc[18500:22200]
#df_info = df_info.iloc[22200:25900]
#df_info = df_info.iloc[25900:29600]
#df_info = df_info.iloc[29600:33300]
#df_info = df_info.iloc[33300:37000]
#df_info = df_info.iloc[37000:40700]
#df_info = df_info.iloc[40700:44400]
#df_info = df_info.iloc[44400:48100]
#df_info = df_info.iloc[48100:51800]
#df_info = df_info.iloc[51800:55429]


person_uid = np.unique(df_info['id_person'])
dict_id = dict(zip(person_uid,np.arange(len(person_uid))))
csi_time = 100 #int(np.max(df_info['len']))
# parameters
max_value = np.max(df_info['max'].values)
#no_classes = len(np.unique(df_info['id_person']))
no_classes = len(dict_id)
csi_subc = 30
input_shape = (csi_time, csi_subc, 6)

# freq BW list
bw_list = pd.read_csv(path_meta+'wifi_f_bw_list_v1.csv')
'''
# avg Array
with open(path_meta + 'dict_avgcsi.pickle','rb') as f:
    dict_avg = pickle.load(f)
'''
# 3D scan param
m,n = 2,3
c =  299792458 # speed of light 
#r = (160 + 160 + 164) * 0.01 # meter
r = 1.64 #meter
d = 45 * 0.01 # meter
ch = 13#3
max_subc = 30

# make data generator
def gen_csi(df_info,id_num,len_num):
    infiles = set([file.replace(".npy","") for file in os.listdir(path_mit_image)])
    tofiles = ['S035_1_1_1'] #list(set(np.unique(df_info.id.values)) - infiles)
    for file in tofiles:    
        # Label
        id_key = df_info[df_info.id==file][['id_person','id_location','id_direction','id_exp']].values[0].astype('int')
        data1_y = dict_id[id_key[0]]

        # read sample data
        # load and uncompress.
        with gzip.open(path_csi+file+'.pickle.gz','rb') as f:
            data1 = pickle.load(f)
        #data1 = data1**4 / (np.abs(data1)**3)
        
        s_idx = (np.arange(len_num) * data1.shape[0] / len_num).astype('int')
        data1_s = data1[s_idx]
        
        #data1_submean = data1[:,0,:,:]
        #data1_submean = np.mean(data1,axis=1)
        #data1_norm = np.array([data1_submean[r] / np.sum(np.abs(data1_submean[r]),axis=1).reshape([2,1]) for r in range(data1_submean.shape[0])])
        # zero pad
        #pad_len = len_num - data1.shape[0]
        #data1_pad = np.pad(data1,((0,pad_len),(0,0),(0,0),(0,0)),'constant',constant_values=0)

        # subcarrier info
        #data1_sc_df = pd.read_csv(path_sc + file + '_df_sc.csv')
        #data1_time = np.diff(data1_sc_df['timestamp_low'])
        #data1_time_pad = np.pad(data1_time,((1,pad_len)),'constant',constant_values=0)

        yield(data1_s ,data1_y,id_key,file)#,data1_time_pad)

gen = gen_csi(df_info,no_classes,csi_time)

target_sig,target_lab,target_id,target_file = next(gen)
th_range,si_range = (30,30)
sig_mat = np.zeros([csi_time,2*th_range,2*si_range])

import time
t1 = time.time()
for idx_th,i in enumerate(range(-th_range,th_range)):
    for idx_si,j in enumerate(range(-si_range,si_range)):
        theta = i * (np.radians(90)/2) / th_range
        sigma = j * (np.radians(90)/2) / si_range
        sig_mat[:,idx_th,idx_si] = Calc3d(target_sig,max_subc,theta+math.pi/2,sigma,bw_list,ch,d,r,c,m,n)
print(time.time()-t1)
#print(f)

f = 'sm_' +str(ch) + '_S035_1_1_1'
ani_frame(sig_mat,f)