# MIT image
# Made by : HERO Kwon
# Date : 190108

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import imageio
import skimage.transform

#Functions
def DenoisePCA(target_xs1,num_eig):
    # remove static
    H_mat = target_xs1 / np.mean(target_xs1,axis=0)
    H_mat = np.nan_to_num(H_mat)

    # eig decomposition
    corr_mat = H_mat.T.dot(H_mat)
    eig_v,eig_w = np.linalg.eig(corr_mat)

    # return num_eig vectors
    return(H_mat.dot(eig_w[1:num_eig+1,:].T))


# data path_mi
path_csi = 'D:\\Data\\Wi-Fi_processed_sm\\'
path_csi_np = 'D:\\Data\\Wi-Fi_processed_npy\\'
path_meta = 'D:\\Data\\Wi-Fi_meta\\'
path_sc = 'D:\\Data\\Wi-Fi_info\\'
path_mit_image = 'G:\\Data\\Wi-Fi_11_mithc'
path_csi_hc = 'D:\\Data\\Wi-Fi_HC\\180_100\\'
path_sig2d = 'D:\\Data\\sig2d_processed\\'


#parameters
target_dir = 1
target_loc = 1
img_shape = (150,250)
pca_num = 50

# data info
df_info = pd.read_csv(path_meta+'data_subc_sig_v1.csv')
target_df = df_info[(df_info.id_location==1) & (df_info.id_direction==1)]

# load data, label
path_file = path_csi_hc
target_data = np.load(path_file + 'Dataset_' + str(target_loc) + '_' + str(target_dir)+'.npy')
target_xs = target_data[:,4:].astype('float32').reshape([-1,500,30*6])
label_csi = target_data[:,:4].astype('int')

# PCA Denoise
pca_xs = np.array([DenoisePCA(target_xs[r],pca_num)  for r in range(target_xs.shape[0])])

# read 2d img
df1 = target_df.iloc[0]
target_labs = df1[['id_person','id_location','id_direction','id_exp']].values
target_idx = np.all(label_csi==target_labs,axis=1)

target_img = imageio.imread(path_sig2d + df1.file)
target_pca = pca_xs[target_idx].squeeze()

img_input = skimage.transform.resize(target_img,img_shape)
csi_input = skimage.transform.resize(target_pca / np.max(np.abs(target_pca)),img_shape)


# data info
df_info = pd.read_csv(path_meta+'data_subc_sig_v1.csv')
df11 = df_info[(df_info.id_location==1) & (df_info.id_direction==1)]
target_df = df11[df11.isin({'id_person': uniq_labels})['id_person']]
uniq_labels = np.intersect1d(np.unique(df_info.id_person),np.unique(label_csi))

target_labs = target_df[['id_person','id_location','id_direction','id_exp']].values
list_pcas = []
list_imgs = []

for i in range(len(target_df)):
    target_idx = np.all(label_csi==target_labs[i],axis=1)
    target_pca = pca_xs[target_idx].squeeze()
    target_img = imageio.imread(path_sig2d + target_df.iloc[i].file)
    list_pcas.append(target_pca)
    list_imgs.append(target_img)