import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import sys
import math
import cv2
import re
%matplotlib inline

path_meta = 'D:\\Data\\Wi-Fi_meta\\'
path_save = 'D:\\Data\\sig2d_bin\\'
path_img = 'D:\\Data\\Wi_Fi Dataset\\Image\\Signature\\'
img_list = os.listdir(path_img)

#import info
pos_img = pd.read_csv(path_meta + 'sig2d_pos_v1.csv')

#parameters
margin = 35
output_shape = (640,360)

#method

def PntsInLine(line1):
    dx = (line1[1,0] - line1[0,0])/5
    dy = (line1[1,1] - line1[0,1])/5
    
    list_pnts = []
    for i in range(6):
        pnts1 = np.array([int(line1[0][0] + i*dx),int(line1[0][1] + i*dy)])
        list_pnts.append(pnts1)
    
    return(np.array(list_pnts))

# main
for file in img_list:
    img_name = re.search('IMG_\d+',file).group(0)
    img_num = int(re.search('\d+',file).group(0))
    # Read the image
    img_read = cv2.imread(path_img + file, 0)

    if (img_read.shape[0] < img_read.shape[1]):
        #flip and reverse
        img = np.fliplr(img_read.T)
    else:
        img = img_read
    # binarize
    #elode,dilate
    kernel = np.ones((3,3),np.uint8)
    e = cv2.erode(img,kernel,iterations = 2)  
    d = cv2.dilate(e,kernel,iterations = 1)

    # Thresholding the image
    (thresh, img_bin) = cv2.threshold(d, 128, 255,cv2.THRESH_BINARY|     cv2.THRESH_OTSU)
    th2 = cv2.adaptiveThreshold(d,255,cv2.ADAPTIVE_THRESH_MEAN_C,\
    cv2.THRESH_BINARY,15,15)
    th3 = cv2.adaptiveThreshold(d,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,\
    cv2.THRESH_BINARY,15,15)
    # Invert the image
    #img_bin = 255-img_bin 
    img_bin = 255-th2
    #cv2.imwrite("Image_bin2.jpg",255-th2)

    # box position
    pos1 = pos_img[pos_img.img==img_num]
        
    if pos1.type.values[0]=='v':    
        #ver
        line1 = pos1.iloc[[0,2]][['pos1','pos2']].values
        line3 = pos1.iloc[[1,3]][['pos1','pos2']].values
        line2 = np.mean([line1,line3],axis=0).astype('int')
    else:
        # hor
        line1 = np.array([img.shape[1] - pos1.iloc[[0,2]]['pos2'].values, pos1.iloc[[0,2]]['pos1'].values]).T
        line3 = np.array([img.shape[1] - pos1.iloc[[1,3]]['pos2'].values, pos1.iloc[[1,3]]['pos1'].values]).T
        line2 = np.mean([line1,line3],axis=0).astype('int')

    pnts1 = PntsInLine(line1)
    pnts2 = PntsInLine(line2)
    pnts3 = PntsInLine(line3)

    for j in range(5):
        img1 = img_bin[pnts1[j,1]+margin:pnts1[j+1,1]-margin,pnts1[j,0]+margin:pnts2[j,0]-margin]
        cv2.imwrite(path_save+img_name+'_'+str(j)+'.png',cv2.resize(img1,output_shape,1,1,interpolation=cv2.INTER_AREA))
        #plt.imshow(img1)
    for k in range(5):
        img1 = img_bin[pnts2[k,1]+margin:pnts2[k+1,1]-margin,pnts2[k,0]+margin:pnts3[k,0]-margin]
        cv2.imwrite(path_save+img_name+'_'+str(k+5)+'.png',cv2.resize(img1,output_shape,1,1,interpolation=cv2.INTER_AREA))
        #plt.imshow(img1)