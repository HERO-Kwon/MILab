import pandas as pd
import numpy as np
import os
import re

# data path_mi
path_meta = 'D:\\Data\\Wi-Fi_meta\\'
path_csi_hc = 'D:\\Data\\Wi-Fi_HC\\180_100\\'
path_sig2d = 'D:\\Data\\sig2d_processed\\'

# load data, label
path_file = path_csi_hc
list_read = []
for filename in os.listdir(path_file):
    list_read.append(np.load(path_file + filename))
    print(filename)

arr_read = np.vstack(list_read)
target_xs = arr_read[:,4:].astype('float32').reshape([-1,500,30,6])
target_lab = arr_read[:,:4].astype('int')

df_lab = pd.DataFrame(target_lab)
df_lab.columns = ['id_person','id_location','id_direction','id_exp']


df_sig = pd.read_csv(path_meta + 'wifi_sig.csv')
img_list = os.listdir(path_sig2d)
df_imglist = pd.DataFrame(img_list,columns=['imgfiles'])
img_ids = [re.match('IMG_\d+',r)[0] for r in img_list]

df_imglist['id_paper'] = img_ids
df_imglist.head()
df_imglist = df_imglist.groupby('id_paper').first()

df_imginfo = pd.merge(df_imglist,df_sig,on='id_paper',how='left')

df_info_sighc = pd.merge(df_lab,df_imginfo,left_on='id_person',right_on='id_sign',how='inner')

arr_uid = np.unique(df_info_sighc['id_person'].values)
dict_idnum = dict(zip(arr_uid,np.arange(len(arr_uid))))
df_info_sighc['uid'] = [dict_idnum[df_info_sighc.iloc[r]['id_person']] for r in range(len(df_info_sighc))]

df_info_sighc.to_csv('df_info_sighc.csv',index=False)