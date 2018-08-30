import scipy.io as sio
import os
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
'''
# For Windows.;.;
dateid = 'csi201807231653'
file_path = '\\\\192.168.10.51\\hdd1tb\\Data\\CSI\\' + dateid
file_list = os.listdir(file_path)
'''
# For Linux
dateid = 'csi201808150100'
file_path = '/home/mint/Drv/HDD1TB/Data/CSI/' + dateid
file_list = os.listdir(file_path)

file_list.sort()
file_list.sort(key=len)

#file_list_1m = file_list[]

# New Processing code

print("CSI:Abs value")
list_10th = [file_list[i] for i in (np.arange(0,1,0.1) * len(file_list)).astype('int')]

df_sc = pd.DataFrame()
dict_csi = {}
scalar_colname = ['timestamp_low','bfee_count','Nrx','Ntx','rssi_a','rssi_b','rssi_c','noise','agc','rate']
for i, file in enumerate(file_list):
    data_read = sio.loadmat(os.path.join(file_path,file))
    num_search = re.search('\d+',file)
    csi_num = int(num_search.group(0))

    # scalar data
    read_sc0_8 = [data_read['csi_entry'][0][0][a][0][0] for a in range(9)]
    data_sc = pd.DataFrame(read_sc0_8).astype('int').T
    data_sc.columns=scalar_colname[0:9]
    data_sc['perm'] = pd.Series([data_read['csi_entry'][0][0][9][0]])
    data_sc['rate'] = data_read['csi_entry'][0][0][10][0][0]
    data_sc.index = [csi_num]

    # csi info
    csi_raw = data_read['csi_entry'][0][0][11]
    csi_scaled = data_read['csi_entry'][0][0][12]

    # aggregate
    df_sc = df_sc.append(data_sc)
    dict_csi[csi_num] = csi_raw,csi_scaled

    if file in list_10th:
        print(int(100*(i+1)/len(file_list)))
'''
# Processing code

print("CSI:Abs value")
list_10th = [file_list[i] for i in (np.arange(0,1,0.1) * len(file_list)).astype('int')]

data_csi = pd.DataFrame()
for i, file in enumerate(file_list):
    data_read = sio.loadmat(os.path.join(file_path,file))
    num_search = re.search('\d+',file)
    csi_num = int(num_search.group(0))
    data_df = pd.DataFrame(data_read['csi_entry'][0],index=[csi_num])
    data_csi = data_csi.append(data_df)
    
    if file in list_10th:
        print(int(100*(i+1)/file_list))

scalar_col = ['timestamp_low','bfee_count','Nrx','Ntx','rssi_a','rssi_b','rssi_c','noise','agc','rate']

data_csi['perm'] = data_csi['perm'].str[0]
for col in scalar_col:
    data_csi[col] = data_csi[col].str[0].str[0]

data_csi = data_csi.sort_index()
'''
ntx = int(np.unique(data_csi.Ntx.values))
nrx = int(np.unique(data_csi.Nrx.values))

# plot1 : abs

print("CSI:to array")
list_abs = []
for i in range(len(dict_csi)):
    abs_val = np.mean(np.abs(dict_csi[i][1]),axis=2)
    #abs_val = np.mean(np.abs(np.angle(dict_csi[i+1][1])),axis=2)
    if abs_val.shape==(ntx,nrx):
        list_abs.append(abs_val)
arr_abs = np.stack(list_abs, axis=-1)

print("CSI: plot")
for i in range(ntx):
    for j in range(nrx):
        plt.plot(df_sc.index,arr_abs[i,j,:],label=str((i,j)))
        plt.legend(bbox_to_anchor=(1.01, 1.01))


# plot2 : subcarrier plot

print("CSI:subcarrier")

max_abs = 30

abs_list = []
ph_list = []
for i in range(len(data_csi)):
    abs_list.append(np.abs(data_csi.iloc[i].csi_scaled))
    ph_list.append(np.angle(data_csi.iloc[i].csi_scaled))
abs_arr = np.array(abs_list)
ph_arr = np.array(ph_list)

abs_int = (abs_arr / max_abs * 255).astype('uint8')
abs_img = np.zeros((abs_int.shape[0],abs_int.shape[3],3),'uint8')
for i in range(abs_int.shape[0]):
    for j in range(abs_int.shape[3]):
        abs_img[i][j] = abs_int[i,0,:,j]

im = Image.fromarray(abs_img)
im.save('im.jpeg')