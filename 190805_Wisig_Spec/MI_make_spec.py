# Made by : HERO Kwon
# Date : 190805

import os
import numpy as np
import pandas as pd
import pickle
import gzip
import matplotlib.pyplot as plt
from scipy import signal
import time

# data path
path_info = 'D:\\Data\\WIFI\\Wi-Fi_meta\\'
path_csi = 'D:\\Data\\WIFI\\Wi-fi_processed\\'
res_path = 'D:\\Data\\WIFI\\Wi-Fi_CARM\\'

# data info
df_sel = pd.read_csv(path_info+'df_subc_sel.csv')

# save res df
col_df = (['version','csi_time','eig_list','nperseg','desc'])

try:
    ver_df = pd.read_csv(res_path + 'ver_df.csv')
except FileNotFoundError:
    ver_df = pd.DataFrame()


#params
version = 'carm_v1'
csi_time = 5500
eig_list = np.arange(1,4)
nperseg=110
desc = ''

def prep_CARM(x,eig_list,nperseg):
    #x = np.abs(data1).reshape([1,-1,30*2*3])
    x_specs = list()
    for i in range(len(x)):
        nn = x[i].T.dot(x[i])
        w,v = np.linalg.eig(nn)
        recon_x = x[i].dot(v[eig_list].T)
        x_spec = list()
        for j in range(len(eig_list)):
            f,t,Sxx = signal.spectrogram(recon_x[:,j],nperseg=nperseg,mode='psd')
            x_spec.append(Sxx)
        x_specs.append(np.array(x_spec).swapaxes(0,1).swapaxes(1,2))
    return(np.array(x_specs))

# data generator
def gen_csi(df1):
    for file in df1.name.values:
        # read sample data
        # load and uncompress.
        with gzip.open(path_csi+file+'.pickle.gz','rb') as f:
            data1 = pickle.load(f)

        # fixed length sampling
        idx_s = np.linspace(0,len(data1)-1,csi_time).astype('int')
        data1_s = data1[idx_s]

        # flatten array for PCA
        flat_data = np.abs(data1_s).reshape([1,-1,30*2*3])
        spec_data = prep_CARM(flat_data,eig_list,nperseg)

        # slicing
        spec_data_sl = spec_data[0,3:56-3,3:56-3,:]

        # label
        lab_data = df1[df1.name==file][['new_id','pos','dir','exp']].values[0]
        
        yield(spec_data_sl,lab_data)

# apply to dataset
for pos in [1,2]:
    for dir in [1,2,3,4]:
        print('job:'+str(pos)+":"+str(dir))
        t1 = time.time()
        list_spec = list()
        list_lab = list()
        for id in np.arange(1,99):
            df1 = df_sel[(df_sel.pos==pos) & (df_sel.dir==dir) & (df_sel.new_id==id)]
            gen = gen_csi(df1)
            list_spec_exp = list()
            for exp in range(10):
                spec,lab = next(gen)
                list_spec_exp.append(spec)
                list_lab.append(lab)
            list_spec.append(np.array(list_spec_exp))
        
        np.save(res_path+version+'_spec_'+str(pos)+'_'+str(dir)+'.npy',np.array(list_spec))
        np.save(res_path+version+'_lab_'+str(pos)+'_'+str(dir)+'.npy',np.array(list_lab))
        print(time.time()-t1)
        
#save results
res_ser =  pd.DataFrame([[version,csi_time,eig_list,nperseg,desc]],columns=col_df)
ver_df = ver_df.append(res_ser,ignore_index=True)  
ver_df.to_csv(res_path + 'ver_df.csv',index=False)