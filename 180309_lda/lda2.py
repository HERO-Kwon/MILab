##########################################
## Made by : HERO Kwon
## Title : LDA 
## Date : 2018.03.09.
## Description : LDA
##########################################

# 2018.03.25 : LDA1 finished
# 2018.03.26~ : LDA2 - cross-val, train-test split

# packages

import numpy as np
import pandas as pd
import scipy as sp
import os
import re
import imageio
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
import random
from sklearn.metrics import confusion_matrix

# import ORLDB images
# image size 56 x 46

array_len = 56*46

file_path = 'D:\Data\ORLDB'

file_list = os.listdir(file_path)
file_list = [s for s in file_list if ".bmp" in s]

image_raw = pd.DataFrame(columns = ['image','person','person_num'])

for file in file_list:
    image_read = imageio.imread(os.path.join(file_path,file),flatten=True)
    [person,person_num] = re.findall('\d\d',file)

    data_read = {'image':image_read,'person':person,'person_num':person_num}
    image_raw = image_raw.append(data_read,ignore_index=True)

# Train-Test Split

image_train = pd.DataFrame()
image_test = pd.DataFrame()

for person in image_raw.person.unique():
    data_train, data_test = train_test_split(image_raw[image_raw.person == person],test_size = 0.5)
    image_train = image_train.append(data_train)
    image_test = image_test.append(data_test)

#image_data = image_train

def LDA_ORLDB(image_data,array_len,num_eigvec):

    img_mat = [np.array(img.reshape(array_len)) for img in image_data.image]
    img_mat = np.vstack(img_mat)
    #img_mat = img_mat.astype('uint16')
    #img_mat = np.asmatrix(img_mat)

    # Computing Mean Vectors

    mean_vectors = []
    for cl in image_data.person.unique():
        mean_vectors.append(np.mean(img_mat[image_data.person == cl],axis=0))

    # Computing Within Scatter matrix

    S_W = np.zeros((array_len,array_len))
    for cl,mv in zip(image_data.person.unique(),mean_vectors):
        class_sc_mat = np.zeros((array_len,array_len))

        for row in img_mat[image_data.person == cl]:
            row, mv = row.reshape(array_len,1), mv.reshape(array_len,1)
            class_sc_mat += (row-mv).dot((row-mv).T)
        S_W += class_sc_mat

    # Computing Between scatter matrix

    overall_mean = np.mean(img_mat,axis=0)

    S_B = np.zeros((array_len,array_len))
    for i,mean_vec in enumerate(mean_vectors):
        n = img_mat[image_data.person==image_data.person.unique()[i]].shape[0]
        mean_vec = mean_vec.reshape(array_len,1)
        overall_mean = overall_mean.reshape(array_len,1)
        S_B += n * (mean_vec - overall_mean).dot((mean_vec - overall_mean).T)

    # Computing Eigenvalue

    eig_vals, eig_vecs = np.linalg.eig(np.linalg.inv(S_W).dot(S_B))

    # sorting eigvectors

    eig_pairs = [(np.abs(eig_vals[i]), eig_vecs[:,i]) for i in range(len(eig_vals))]
    eig_pairs = sorted(eig_pairs,key=lambda k: k[0], reverse=True)

    # Choosing Eigvalue
    #W = np.hstack((eig_pairs[0][1].reshape(array_len,1),eig_pairs[1][1].reshape(array_len,1)))
    #W = eig_vecs[0:20,:].T
    # Transform to new subspace
    #img_lda = img_mat.dot(W)

    W = np.array([eig_pairs[i][1] for i in range(num_eigvec)])

    return(W)


#lda_train = LDA_ORLDB(image_train,array_len)
#lda_test = LDA_ORLDB(image_test,array_len)

w_train = LDA_ORLDB(image_train,array_len,100)

mat_train = [np.array(img.reshape(array_len)) for img in image_train.image]
mat_train = np.vstack(mat_train)
mat_test = [np.array(img.reshape(array_len)) for img in image_test.image]
mat_test = np.vstack(mat_test)

lda_train = mat_train.dot(w_train.T)
lda_test = mat_test.dot(w_train.T)


## Error Matrix

mat_err = np.zeros((lda_test.shape[0],lda_test.shape[0]))
dict_err = {}

for i in range(lda_test.shape[0]):
    for j in range(lda_test.shape[0]):
        dist_err = np.linalg.norm(lda_test[i] - lda_test[j])
        mat_err[i,j] = dist_err
        key_err = ((image_test.person.iloc[i],image_test.person.iloc[j]),image_test.person_num.iloc[j])
        if i!=j : dict_err[key_err] = dist_err


# Distribution Curve

person_comp = [(person[0],person[1]) for person,person_num in list(dict_err.keys())]

person_tf = [person[0]==person[1] for person in person_comp]

dist_true = [list(dict_err.values())[i] for i in range(len(dict_err)) if person_tf[i]==True]
dist_false = [list(dict_err.values())[i] for i in range(len(dict_err)) if person_tf[i]==False]


plt.hist(dist_true,bins='auto',normed=1,histtype='step',color='blue',label='Dist_True')
plt.hist(dist_false,bins='auto',normed=1,histtype='step',color='red',label='Dist_False')
plt.legend(loc='upper right')
plt.show()

## EER Curve

eer_df = pd.DataFrame(columns = ['thres','fn','fp'])
#err_values = random.sample(list(dict_err.values()),1000)
err_values = dict_err.values()

real_tf = [e[0]==e[1] for e in person_comp]
for i, thres in enumerate(set(err_values)):
    predicted_tf = [e <= thres for e in dict_err.values()]
    
    tn, fp, fn, tp = confusion_matrix(real_tf,predicted_tf).ravel()

    eer_ser = {'thres':thres,'tn':tn,'fp':fp,'fn':fn,'tp':tp}
    eer_df = eer_df.append(eer_ser,ignore_index=True)
    
    curr_percent = 100 * (i+1) / len(set(err_values))
    if (curr_percent % 10)==0 : print(curr_percent)

eer_df_graph = eer_df.sort_values(['thres'])
eer_df_graph.fn = eer_df_graph.fn / max(eer_df_graph.fn) * 100
eer_df_graph.fp = eer_df_graph.fp / max(eer_df_graph.fp) * 100
eer_df_graph.te = eer_df_graph.fn + eer_df_graph.fp

min_te_pnt = eer_df_graph[eer_df_graph.te == min(eer_df_graph.te)]
min_te_val = float((min_te_pnt['fn'].values + min_te_pnt['fp'].values) / 2)

plt.plot(eer_df_graph.thres,eer_df_graph.fn,color='red',label='FNR')
plt.plot(eer_df_graph.thres,eer_df_graph.fp,color='blue',label='FPR')
plt.plot(eer_df_graph.thres,eer_df_graph.te,color='green',label='TER')
plt.axhline(min_te_val,color='black')
plt.text(max(eer_df_graph.thres)*0.8,min_te_val-5,'EER : ' + str(round(min_te_val,2)))
plt.legend()
plt.show()