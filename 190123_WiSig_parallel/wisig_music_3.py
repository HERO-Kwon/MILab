import numpy as np
import pandas as pd
import os
import gzip,pickle
import math
import cmath
import seaborn as sns
import matplotlib.pyplot as plt
import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D
import numba
from numba import jit

path_csi = 'G:\\Data\\Wi-Fi_processed\\'
path_meta = 'D:\\Data\\Wi-Fi_meta\\'
path_save = 'G:\\Data\\wisig_music\\'
path_csinp = 'G:\\Data\\wisig_np\\'

#select data
#data_idx = np.arange(0,256)
#data_idx = np.arange(256,512)
data_idx = np.arange(512,768)
#data_idx = np.arange(768,1024)
#data_idx = np.arange(1024,1283)



# freq BW list
bw_list = pd.read_csv(path_meta+'wifi_f_bw_list_v1.csv')

# parameters
# parameters
d = 0.45
c = 300 * 10**6
ch=7
f_d = 20 * 10**6 / 30


@jit
def Sigma(theta,d,f,c):
    return cmath.exp(-1j * 2 * math.pi * d * math.sin(theta) * f / c)
@jit
def Omega(tau,f_d):
    return cmath.exp(-1j * 2 * math.pi * f_d * tau)
@jit
def SteerVec(theta,tau):
    list_a = []
    for rx in range(2):
        for subc in range(15):
            f = bw_list[str(subc)][ch-1] * 10**6
            list_a.append(Sigma(theta,d,f,c)**rx * Omega(tau,f_d)**subc)
    arr_a = np.array(list_a).reshape([-1,1])
    return(arr_a) #np.hstack([arr_a]*30))
@jit
def Pmu(mat_en1,theta,tau):
    mat_a = SteerVec(theta,tau)
    return(1 / np.transpose(np.conjugate(mat_a)).dot(mat_en1).dot(np.transpose(np.conjugate(mat_en1))).dot(mat_a))
@jit
def SmoothCSI(dt_csi):
    list_rxs = []
    for rx in range(3):
        list_rx1 = []
        for subc in range(15):
            list_rx1.append(dt[subc:subc+15,rx])
        list_rxs.append(np.array(list_rx1))

    list_smcsi = []
    for i in range(2):
        csi_arr1 = np.hstack([list_rxs[i],list_rxs[i+1]])
        list_smcsi.append(csi_arr1)
    return(np.vstack([list_smcsi[0],list_smcsi[1]]))
@jit
def NoiseVec(mat_x):
    mat_xx = mat_x.dot(np.transpose(np.conjugate(mat_x)))
    eigv, mat_en = np.linalg.eig(mat_xx)
    mat_en1 = mat_en[np.argmin(eigv),:].reshape([-1,1])
    return(mat_en1)

read_csi = np.load(path_csinp + 'csi_1_1.npy')
read_lab = np.load(path_csinp + 'csi_1_1_labels.npy')

import time
for idx in data_idx:
    t1 = time.time()
    
    data1 = read_csi[idx]
    label1 = read_lab[idx]
    
    mat_music = np.zeros([500,2,10,10]).astype('float32')
    for tx in range(2):
        for t in range(500):
            dt = data1[t,:,tx,:]

            # make music array
            #dt = csi_tx1[1000]
            mat_x = SmoothCSI(dt)
            mat_en1 = NoiseVec(mat_x)

            # th, tau range

            th_arr = np.linspace(-math.pi/6,math.pi/6,10)
            ta_arr = np.linspace(0,10,10)

            # calc music

            for i,ta in enumerate(ta_arr):
                for j,th in enumerate(th_arr):
                    mat_music[t,tx,i,j] = 20*math.log10(np.abs(Pmu(mat_en1,th,ta)))
                    
    # save file
    save_name = str(label1[0]) + '_' + str(label1[1]) + '_' + str(label1[2]) + '_' + str(label1[3]) + '.npy'
    np.save(path_save + save_name, mat_music)
    
    print(time.time() - t1)
    