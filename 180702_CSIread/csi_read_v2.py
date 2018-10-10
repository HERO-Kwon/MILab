import scipy.io as sio
import os
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image

# For Windows.;.;
dateid = 'csi201809112003'
file_path = 'D:\\CSI\\' + dateid
file_list = os.listdir(file_path)
label_path = 'D:\\imacros'
label_list = os.listdir(label_path)
label_list = [f for f in label_list if f.endswith('.csv')]
phone_list = pd.read_csv('D:\\known_phone_list.csv')

# Label Processing
datepeople_df = pd.DataFrame()
for i, label_file in enumerate(label_list):
    match_dt = re.match("Extract_(?P<day>\d\d)(?P<month>\d\d)(?P<year>\d\d)_(?P<hour>\d\d)(?P<minute>\d\d)(?P<second>\d\d)\.csv",label_file)
    datetimes = match_dt.groupdict()
    datetimes = dict((k, int(v)) for k, v in datetimes.items())
    datetimes['year'] += 2000
    datetimes_ser = pd.Series(datetimes)

    read_df = pd.read_csv(label_path+'\\'+label_file)
    people_ser = pd.Series()
    #ff = re.match(phone_list.loc[3].values[0],f.loc[9].values[0])
    for j in range(len(phone_list)):
        mac_addr = phone_list.loc[j].mac
        mac_name = phone_list.loc[j].owner
        people_ser[mac_name] = len(read_df[read_df['내부 네트워크 정보'].str.contains(mac_addr)])
    people_ser['total_people'] = sum(people_ser)
    datepeople_ser = pd.concat((datetimes_ser,people_ser))
    datepeople_df = datepeople_df.append(datepeople_ser,ignore_index=True)
datepeople_df = datepeople_df.astype('int')
col_sequence = ['year','month','day','hour','minute','second','total_people','HERO','Zaynab','SI','JY','JS','SC']
datepeople_df = datepeople_df.reindex(columns=col_sequence)

# CSI Processing

print("CSI:Abs value")

file_list.sort()
file_list.sort(key=len)
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

ntx = int(np.unique(data_sc.Ntx.values))
nrx = int(np.unique(data_sc.Nrx.values))
# plot2 : by subcarrier
print("CSI:to array")
list_abs2 = []
for i in range(len(dict_csi)):
    abs_val = np.reshape(np.abs(dict_csi[i+1][1]),((1,90)))
    #abs_val = np.mean(np.abs(np.angle(dict_csi[i+1][1])),axis=2)
    if abs_val.shape==(ntx,nrx*30):
        list_abs2.append(abs_val)
arr_abs2 = np.stack(list_abs2, axis=-1)

a=np.squeeze(arr_abs2)
plt.plot(a)
plt.show()
# plot1 : abs

print("CSI:to array")
list_abs = []
for i in range(len(dict_csi)):
    abs_val = np.mean(np.abs(dict_csi[i+1][1]),axis=2)
    #abs_val = np.mean(np.abs(np.angle(dict_csi[i+1][1])),axis=2)
    if abs_val.shape==(ntx,nrx):
        list_abs.append(abs_val)
arr_abs = np.stack(list_abs, axis=-1)

print("CSI: plot")
for i in range(ntx):
    for j in range(nrx):
        plt.plot(df_sc.index,arr_abs[i,j,:],label=str((i,j)))
        plt.legend(bbox_to_anchor=(1.01, 1.01))



# plot3 : subcarrier plot

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