#%matplotlib notebook

import scipy.io as sio
import os
import re
import numpy as np
import pandas as pd
from PIL import Image
import pickle
import gzip
from plots_csi_sign import *
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.animation import FFMpegWriter
#plt.rcParams['animation.ffmpeg_path'] ='C:\\ffmpeg-4.1-win64-static\\bin\\ffmpeg.exe'



# helper function
def search_int_in_filename(filename):
    num_search = re.search('\d+',filename)
    return int(num_search.group(0))


folder_path = 'D:\Data\Wi_Fi Dataset\Wi-Fi dataset_CSI\\'
folder_list = os.listdir(folder_path)
save_path = 'D:\Data\Wi_Fi Dataset\Wi-Fi_processed_sm\\'
#file_finished = [re.search('S\d+_\d+_\d+_\d+',row).group(0) for row in os.listdir(save_path)]

#folder_list = folder_list[0:826]
#folder_list = folder_list[826:1652]
#folder_list = folder_list[1652:2478]
#folder_list = folder_list[2478:3304]
#folder_list = folder_list[3304:4130]
#folder_list = folder_list[4130:4956]
folder_list = folder_list[4956:5782]
#folder_list = folder_list[5782:6608]
#folder_list = folder_list[6608:7434]
#folder_list = folder_list[7434:len(folder_list)]

df_subc = pd.DataFrame()
dict_scaled = {}
for exp_id in folder_list:
    file_path = folder_path + exp_id
    file_list = os.listdir(file_path)
    file_list.sort(key=search_int_in_filename)
        
    df_sc = pd.DataFrame()
    list_scaled = []
    scalar_colname = ['timestamp_low','bfee_count','Nrx','Ntx','rssi_a','rssi_b','rssi_c','noise','agc','rate']

    for i, file in enumerate(file_list):
        data_read = sio.loadmat(os.path.join(file_path,file))
        num_search = re.search('\d+',file)
        csi_num = int(num_search.group(0))
        '''
        # scalar data
        read_sc0_8 = [data_read['csi_entry'][0][0][a][0][0] for a in range(9)]
        data_sc = pd.DataFrame(read_sc0_8).astype('int').T
        data_sc.columns=scalar_colname[0:9]
        data_sc['perm'] = pd.Series([data_read['csi_entry'][0][0][9][0]])
        data_sc['rate'] = data_read['csi_entry'][0][0][10][0][0]
        data_sc.index = [(exp_id,csi_num)]
        '''
        # csi info
        csi_sm = data_read['csi_entry'][0][0][11]
        #csi_scaled = data_read['csi_entry'][0][0][12]

        # aggregate
        df_sc = df_sc.append(data_sc)
        list_scaled.append(csi_sm)
    #df_sc.to_csv(save_path+exp_id+'_df_sc.csv')
    try:
        arr_scaled = np.array(list_scaled).reshape(-1,30,2,3)
    except:
        pass
    #dict_scaled[exp_id] = arr_scaled

    #Data Save
    with gzip.open(save_path+exp_id+'.pickle.gz', 'wb') as f:
        pickle.dump(arr_scaled, f, pickle.HIGHEST_PROTOCOL)
    '''
    #animated plot
    plot_animated(save_path,exp_id,arr_scaled)
    #heatmap
    fig_len = len(arr_scaled)/50
    for t in range(2):
        for r in range(3):
            arr_abs = np.abs(arr_scaled[:,:,t,r])
            arr_ph = np.cos(np.angle(arr_scaled[:,:,t,r]))
            
            heatmap_array(save_path,(10,fig_len),exp_id+'_abs'+str((t,r)),arr_abs,50,0)
            heatmap_array(save_path,(10,fig_len),exp_id+'_ph'+str((t,r)),arr_ph,np.pi/2,-np.pi/2)
    '''
    # calc mean, var
    ser_subc = pd.Series(name=exp_id)
    mean_subc = np.mean(np.abs(arr_scaled))
    std_subc = np.std(np.abs(arr_scaled))
    len_subc = len(arr_scaled)

    ser_subc['mean'] = mean_subc
    ser_subc['std'] = std_subc
    ser_subc['len'] = len_subc

    df_subc = df_subc.append(ser_subc)
    df_subc.to_csv(save_path+exp_id+'_df_subc.csv')

#with gzip.open(save_path+exp_id+'.pickle.gz', 'wb') as f:
#    pickle.dump(dict_scaled, f, pickle.HIGHEST_PROTOCOL)