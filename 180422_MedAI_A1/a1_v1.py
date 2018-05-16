import numpy as np
import os
import scipy.io as sio
import re

# For Windows
file_path_fs = 'D:\Matlab_Drive\Data\MedAI\MAI_Project1\Train_image_label' # im_org - full sampled
file_path_ksps = 'D:\Matlab_Drive\Data\MedAI\MAI_Project1\Train_k'   # f_im - ksp undersampling
file_path_unds = 'D:\Matlab_Drive\Data\MedAI\MAI_Project1\Train_image_input' # im_u - undersampled



# Read Full Sampled Data
data_fs = {}
file_path = file_path_fs

file_list = os.listdir(file_path)
file_list = [s for s in file_list if ".mat" in s]

for file in file_list:

    os.path.join(file_path,file)
    data_read = sio.loadmat(os.path.join(file_path,file))

    data_arr = data_read['im_org']
    data_lab = re.findall('\d+',file)

    data_fs[data_lab[0]] = data_arr


# Read Under Sampled Data
data_unds = {}
file_path = file_path_unds

file_list = os.listdir(file_path)
file_list = [s for s in file_list if ".mat" in s]

for file in file_list:

    os.path.join(file_path,file)
    data_read = sio.loadmat(os.path.join(file_path,file))

    data_arr = data_read['im_u']
    data_lab = re.findall('\d+',file)

    data_unds[data_lab[0]] = data_arr