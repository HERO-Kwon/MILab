import os
import re
import pandas as pd
import numpy as np

# path
path_sign = 'J:\\Data\\sig2d_processed\\'
list_sign = os.listdir(path_sign)

#make sig2d info
wifi_sig = pd.read_csv('J:\\Data\\wifi_sig.csv')
list_sign_id = [r[:8] for r in list_sign]
df_sign  = pd.DataFrame({'id_paper':list_sign_id,'file':list_sign})
info_sig2d = pd.merge(df_sign,wifi_sig,how='left')
info_sig2d.to_csv('info_sig2d.csv')
info_sig2d.head()

#extract id from subc
df_subc = pd.read_csv('J:\\Data\\df_subc.csv')
df_subc.columns = ['id','len','mean','std']
split_id = df_subc['id'].str.split(pat='_',expand=True)
id_person_num = [int(re.findall('\d+',r)[0]) for r in split_id[0].values]
df_subc['id_person'] = id_person_num
df_subc['id_location'] = split_id[1]
df_subc['id_direction'] = split_id[2]
df_subc['id_exp'] = split_id[3]
df_subc.head()

# select data
# filter outlier
len_min = np.mean(df_subc['len']) - np.std(df_subc['len'])
len_max = np.mean(df_subc['len']) + np.std(df_subc['len'])
df_subc_islen = df_subc[(df_subc.len >= len_min) & (df_subc.len <= len_max)]
df_subc_isimg = df_subc[df_subc['id_person'].isin(info_sig2d.id_sign)]
data_subc = df_subc.loc[set(df_subc_islen.index) & set(df_subc_isimg.index)]
data_subc_sig = pd.merge(data_subc,info_sig2d,how='left',left_on='id_person',right_on='id_sign')
data_subc_sig.to_csv('data_subc_sig.csv',index=None)
data_subc_sig.head()

#plotting
from matplotlib.pyplot import cm
data_plot = data_subc[(data_subc.id_location=='1') & (data_subc.id_direction=='1')]
X_plot = data_plot[['mean','std']].values.astype('float')
#categories = yolo_cat.supercategory
categories = data_plot['id_person'].values
cat = set(categories)
target_ids = range(len(cat))

color=cm.tab20(np.linspace(0,1,len(cat)))
from matplotlib import pyplot as plt
plt.figure()
for i, c, label in zip(target_ids, color, cat):
    plt.scatter(X_plot[categories==label, 0], X_plot[categories==label, 1],c=c, label=label)
plt.legend()
plt.show()


