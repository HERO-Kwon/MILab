import scipy.io as sio
import os
import re
import numpy as np
import matplotlib.pyplot as plt
'''
# For Windows
dateid = 'csi201807231653'
file_path = '\\\\192.168.10.51\\hdd1tb\\Data\\CSI\\' + dateid
file_list = os.listdir(file_path)
'''
# For Linux
dateid = 'csi201807282057'
file_path = '/home/mint/Drv/HDD1TB/Data/CSI/' + dateid
file_list = os.listdir(file_path)

print("CSI:Abs value")
abs_csi = {}
for file in file_list:
    data_read = sio.loadmat(os.path.join(file_path,file))
    num_search = re.search('\d+',file)
    csi_num = num_search.group(0)
    abs_csi[csi_num] = np.abs(data_read['csi_entry']['csi_scaled'][0][0])

print("CSI:to array")
list_abs = []
for i in range(len(abs_csi)):
    abs_val = np.mean(abs_csi[str(i+1)],axis=2)
    if abs_val.shape==(1,3):
        list_abs.append(abs_val)
arr_abs = np.stack(list_abs, axis=-1)

print("CSI: plot")
for i in range(1):
    for j in range(3):
        plt.plot(arr_abs[i,j,:],label=str((i,j)))
        plt.legend(bbox_to_anchor=(1.01, 1.01))
