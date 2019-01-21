# MIT image : Analysis
# Made by : HERO Kwon
# Date : 190113
from numba import jit
import os
import numpy as np
import pandas as pd
import pickle
import matplotlib.pyplot as plt
#import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import StratifiedKFold
import time

# data path
#path_meta = '/home/herokwon/mount_data/Data/Wi-Fi_meta/'
#path_csi = '/home/herokwon/mount_data/Data/Wi-Fi_processed/'
#path_csi_np = '/home/herokwon/mount_data/Data/Wi-Fi_processed_npy/'
#path_mit_image = '/home/herokwon/mount_data/Data/Wi-Fi_mit_image/'

path_csi = 'D:\\Data\\Wi-Fi_processed\\'
path_csi_np = 'D:\\Data\\Wi-Fi_processed_npy\\'
path_meta = 'D:\\Data\\Wi-Fi_meta\\'
path_sc = 'D:\\Data\\Wi-Fi_info\\'
path_mit_image = 'D:\\Data\\Wi-Fi_mit_abs12\\'
path_movie = 'D:\\Data\\Wi-Fi_movie\\'

# data info
df_info = pd.read_csv(path_meta+'data_subc_sig_v1.csv')
#df_info = df_info[df_info.id_location==1 ]

person_uid = np.unique(df_info['id_person'])
dict_id = dict(zip(person_uid,np.arange(len(person_uid))))
csi_time = 100 #15000 #int(np.max(df_info['len']))
# parameters
max_value = np.max(df_info['max'].values)
#no_classes = len(np.unique(df_info['id_person']))
no_classes = len(dict_id)
csi_subc = 30
input_shape = (csi_time, csi_subc, 6)

# freq BW list
bw_list = pd.read_csv(path_meta+'wifi_f_bw_list.csv')

# 3D scan param
m,n = 2,3
c =  299792458 # speed of light 
#r = (160 + 160 + 164) * 0.01 # meter
r = 1.64 #meter
d = 45 * 0.01 # meter
ch = 8#3
max_subc = 30

# Load data
saved_images = [f.replace(".npy","") for f in os.listdir(path_mit_image)]
saved_images

df11 = df_info[(df_info.id_direction==1) & (df_info.id_location==1)]
df_target = df11[df11['id'].isin(saved_images)]
df_lab = df_target[['id','id_person','id_location','id_direction','id_exp']]
df_lab = df_lab.drop_duplicates()
df_lab

data_mit = []
filename_mit = []
for file in df_lab.id.values:
    data_load = np.load(path_mit_image + file + '.npy')
    dl_norm = np.array([data_load[i] / np.max(data_load[i]) for i in range(data_load.shape[0])])
    data_mit.append(dl_norm)
    filename_mit.append(file)


label_mit = df_lab.id_person.values
file_mit = df_lab.id.values

arr_mit = np.array(data_mit)
#diff_mit = np.diff(arr_mit,axis=1)
sum_mit = arr_mit.reshape([-1,100*10*10])#np.sum(diff_mit,axis=1).reshape([-1,10*10])
norm_mit = sum_mit

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
    ani = animation.FuncAnimation(fig,update_img,100)#,interval=1)
    writer = animation.writers['ffmpeg'](fps=10)

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
    fig1.savefig(path_movie+file+'_sum.png')
    
    plt.close('all')
    return ani

import matplotlib.animation as animation
import numpy as np
from pylab import *

dpi = 100

for i,file in enumerate(file_mit):
    ani_frame(arr_mit[i,:,:,:],file)
    