import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import skimage.data as data
import skimage.feature as feature
import skimage.transform as transform

# for truncated img files
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

#image path
path_img = 'D:\\Data\\AnimalsOnTheWeb_v1\\'

#df to aggregate results
lbp_df = pd.DataFrame()

list_folder = os.listdir(path_img)
list_folder = [f for f in list_folder if not 'other' in f]

for foldername in list_folder:
    print(foldername)
    list_files = os.listdir(path_img+foldername+'\\')
    list_files = [f for f in list_files if 'pic' in f]

    for filename in list_files:
        try:
            img_arr = data.load(path_img + foldername+'\\'+filename,as_grey=True)
        
            lbp = feature.local_binary_pattern(img_arr, 8, 2, "uniform") #image, P, R, method
            brickCount = 10#int(lbp.max() + 1)
            lbp_hist, _ = np.histogram(lbp, normed = True, bins = brickCount, range = (0, brickCount))

            img_col = data.load(path_img + foldername+'\\'+filename,as_grey=False)
            width,height,channel= img_col.shape
            c_height = height // 2
            c_width = width // 2
            mean_col = []
            for i in range(channel):
                img0 = img_col[:c_width,:c_height,i]
                img1 = img_col[c_width:,:c_height,i]
                img2 = img_col[:c_width,c_height:,i]
                img3 = img_col[c_width:,c_height:,i]
                
                mean_col.append(np.array([np.mean(img0),np.mean(img1),np.mean(img2),np.mean(img3)]))
            arr_mcol = np.array(mean_col).flatten()

        except:
            lbp_hist = np.array([0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0])

        # Result
        lbp_ser = pd.Series([foldername,filename,\
                lbp_hist[0],lbp_hist[1],lbp_hist[2],lbp_hist[3],lbp_hist[4],\
                lbp_hist[5],lbp_hist[6],lbp_hist[7],lbp_hist[8],lbp_hist[9],\
                arr_mcol[0],arr_mcol[1],arr_mcol[2],arr_mcol[3],arr_mcol[4],\
                arr_mcol[5],arr_mcol[6],arr_mcol[7],arr_mcol[8],arr_mcol[9],\
                arr_mcol[10],arr_mcol[11]],
                index=['Animal','File','LBP0','LBP1','LBP2','LBP3','LBP4',\
                        'LBP5','LBP6','LBP7','LBP8','LBP9',\
                        'cr0','cr1','cr2','cr3','cg0','cg1','cg2','cg3',\
                        'cb0','cb1','cb2','cb3'])
        # Aggregate result
        lbp_df.append(lbp_ser,ignore_index=True)


lbp_df.to_csv('Res_LBP_color.csv')