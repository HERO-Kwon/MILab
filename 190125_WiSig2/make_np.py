import pandas as pd
import numpy as np
import os
import re
import matplotlib.pyplot as plt
import pickle
import gzip
import math
import scipy.signal
import re
# data path_mi
path_meta = 'D:\\Data\\Wi-Fi_meta\\'
path_csi_hc = 'D:\\Data\\Wi-Fi_HC\\180_100\\'
path_sig2d = 'D:\\Data\\sig2d_processed\\'
path_csi = 'G:\\Data\\Wi-Fi_processed\\'
path_csism = 'D:\\Data\\Wi-Fi_processed_sm\\'
path_csi_np = 'G:\\Data\\wisig_np\\'

def SamplingHC(data1,n_sample):
    data1_hc = np.zeros([500,30,2,3]).astype('complex64')
    for i in range(data1.shape[1]):
        for j in range(data1.shape[2]):
            for k in range(data1.shape[3]):
                s_idx = (np.arange(n_sample) * data1.shape[0] / n_sample).astype('int')
                data1_s = data1[s_idx,i,j,k]
                #data1_s = scipy.signal.resample(data1_a[:,i,j,k], 500, None)
                data1_hc[:,i,j,k] = data1_s
    return(data1_hc)


n_samples = 500
files = os.listdir(path_csi)

list_lab = []
list_csi = []
for i,file in enumerate(files):
    print((i+1) / len(files))
    try:
        with gzip.open(path_csi+file,'rb') as f:
            data1 = pickle.load(f)
        list_csi.append(SamplingHC(data1,n_samples))
        lab_numbers = re.findall('\d+',file)
        list_lab.append(lab_numbers)
    except:
        pass