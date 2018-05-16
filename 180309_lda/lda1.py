##########################################
## Made by : HERO Kwon
## Title : LDA 
## Date : 2018.03.09.
## Description : LDA
##########################################


# packages

import numpy as np
import pandas as pd
import scipy as sp
import os
import re
import imageio
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split

# import ORLDB images
# image size 56 x 46

file_path = 'D:\Data\ORLDB'

file_list = os.listdir(file_path)
file_list = [s for s in file_list if ".bmp" in s]

image_data = pd.DataFrame(columns = ['image','person','person_num'])

for file in file_list:
    image_read = imageio.imread(os.path.join(file_path,file),flatten=True)
    [person,person_num] = re.findall('\d\d',file)

    data_read = {'image':image_read,'person':person,'person_num':person_num}
    image_data = image_data.append(data_read,ignore_index=True)


img_mat = [np.array(img.reshape(56*46)) for img in image_data.image]
img_mat = np.vstack(img_mat)

# Train-Test Split

image_train = pd.DataFrame()
image_test = pd.DataFrame()

for person in image_data.person.unique():
    data_train, data_test = train_test_split(image_data[image_data.person == person],test_size = 0.5)
    image_train = image_train.append(data_train)
    image_test = image_test.append(data_test)


# Computing Mean Vectors

mean_vectors = []
for cl in image_data.person.unique():
    mean_vectors.append(np.mean(img_mat[image_data.person == cl],axis=0))

# Computing Within Scatter matrix

S_W = np.zeros((56*46,56*46))
for cl,mv in zip(image_data.person.unique(),mean_vectors):
    class_sc_mat = np.zeros((56*46,56*46))

    for row in img_mat[image_data.person == cl]:
        row, mv = row.reshape(56*46,1), mv.reshape(56*46,1)
        class_sc_mat += (row-mv).dot((row-mv).T)
    S_W += class_sc_mat

# Computing Between scatter matrix

overall_mean = np.mean(img_mat,axis=0)

S_B = np.zeros((56*46,56*46))
for i,mean_vec in enumerate(mean_vectors):
    n = img_mat[image_data.person==image_data.person.unique()[i]].shape[0]
    mean_vec = mean_vec.reshape(56*46,1)
    overall_mean = overall_mean.reshape(56*46,1)
    S_B = n * (mean_vec - overall_mean).dot((mean_vec - overall_mean).T)

# Computing Eigenvalue

eig_vals, eig_vecs = np.linalg.eig(np.linalg.inv(S_W).dot(S_B))

for i in range(len(eig_vals)):
    eigvec_sc = eig_vecs[:,i].reshape(56*46,1)

# sorting eigvectors

eig_pairs = [(np.abs(eig_vals[i]), eig_vecs[:,i]) for i in range(len(eig_vals))]
eig_pairs = sorted(eig_pairs,key=lambda k: k[0], reverse=True)

# Choosing Eigvalue
W = np.hstack((eig_pairs[0][1].reshape(56*46,1),eig_pairs[1][1].reshape(56*46,1)))

# Transform to new subspace
img_lda = img_mat.dot(W)

# plot
# plt.scatter(x=img_lda[:,0].real,y=img_lda[:,1].real)

for label in image_data.person.unique():

    plt.scatter(x=img_lda[:,0].real[image_data.person == label],
    y = img_lda[:,1].real[image_data.person == label],
    label=label
    )

