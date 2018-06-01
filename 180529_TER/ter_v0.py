##########################################
## Made by : HERO Kwon
## Title : TER
## Version : v0
## Date : 2018.05.29.
## Description : TER-RM Algorithm
##########################################


# Main

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
import time

from sklearn.datasets import load_iris
 
data = load_iris()
data_v = data['data']
data_l = data['target']

data_train = np.array([]).reshape(0,4)
data_test = np.array([]).reshape(0,4)
label_train = np.array([])
label_test = np.array([])

for target in list(set(data['target'])):
    v_train, v_test = train_test_split(data_v[data_l==target],test_size = 0.5)
    l_train = np.full(shape=len(v_train),fill_value=target)
    l_test = np.full(shape=len(v_test),fill_value=target)
    data_train = np.concatenate((data_train,v_train))
    data_test = np.concatenate((data_test,v_test))
    label_train = np.concatenate((label_train,l_train))
    label_test = np.concatenate((label_test,l_test))

## TER Algorithm
# Basis : RM2
def RMmodel(order,X):
    m,l = X.shape
    
    M1 = []
    M2 = []
    M3 = []
    MM1 = []
    MM3 = []

    Msum = np.sum(X,axis=1)

    for i in range(order):
        for k in range(l):
            M1.append(X[:,k]**(i+1))
            if (i>0):
                M3.append(X[:,k]*Msum**(i)) 
        M2.append(Msum**(i+1))
        MM1.append(M1)
        if (i>0):
            MM3.append(M3)

    MM1 = np.array(MM1).T
    MM1 = MM1.reshape((m,-1,1)).squeeze(axis=2)
    M2 = np.array(M2).T
    if (len(MM3)):
        MM3 = np.array(MM3).T
        MM3 = MM3.reshape((m,-1,1)).squeeze(axis=2)
        P = np.concatenate((np.ones((m,1)),MM1,M2,MM3),axis=1)
    else : P = np.concatenate((np.ones((m,1)),MM1,M2),axis=1)

    return(P)

def TERmodel(rank,r,n,X,Y):
    alpha = []
    for k in list(set(Y)):

        P_n = RMmodel(rank,X[Y!=k])
        P_p = RMmodel(rank,X[Y==k])

        mk_n = X[Y!=k].shape[0]
        mk_p = X[Y==k].shape[0]

        yk_n = (r-n) * np.ones(shape=Y[Y!=k].shape)
        yk_p = (r+n) * np.ones(shape=Y[Y==k].shape)

        I = np.eye(P_n.shape[1])
        b = 10**(-4)

        first_eq = np.linalg.pinv(b*I + (1/mk_n)*(P_n.T).dot(P_n) + (1/mk_p)*(P_p.T).dot(P_p))
        second_eq = (1/mk_n)*(P_n.T).dot(yk_n) + (1/mk_p)*(P_p.T).dot(yk_p)
        ak = np.dot(first_eq,second_eq)

        alpha.append(ak)
    return(np.array(alpha).T)

alpha = TERmodel(6,0.5,0.5,data_train,label_train)

P_t = RMmodel(6,data_test)
yt = P_t.dot(alpha)
yt1 = np.argmax(yt,axis=1)
plt.plot(yt1)

# Training
''' 
# RM training
P = RMmodel(6,data_train)
I = np.eye(P.shape[1])
b = 10**(-4)

alpha = np.linalg.pinv((P.T).dot(P)+b*I).dot(P.T).dot(label_train)

# Testing
P = RMmodel(6,data_test)
I = np.eye(P.shape[1])
b = 10**(-4)
alpha = np.linalg.pinv((P.T).dot(P)+b*I).dot(P.T).dot(label_test)

y_train = P.dot(alpha)

'''


'''
# Functions
## Functiobn : LDA
def LDA_ORLDB(image_data,array_len,num_eigvec):

    print("LDA Calculation")

    img_mat = [np.array(img.reshape(1,array_len)) for img in image_data.image]
    img_mat = np.vstack(img_mat)

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
        
        cls_prob = img_mat[image_data.person==cl].shape[0] / img_mat.shape[0]
        S_W += cls_prob * class_sc_mat

    # Computing Between scatter matrix

    overall_mean = np.mean(img_mat,axis=0)

    S_B = np.zeros((array_len,array_len))
    for mean_vec in mean_vectors:
        mean_vec = mean_vec.reshape(array_len,1)
        overall_mean = overall_mean.reshape(array_len,1)
        S_B += (mean_vec - overall_mean).dot((mean_vec - overall_mean).T)

    # Computing Eigenvalue

    eig_vals, eig_vecs = np.linalg.eig(np.linalg.pinv(S_W).dot(S_B))

    # sorting eigvectors

    eig_pairs = [(np.abs(eig_vals[i]), eig_vecs[:,i]) for i in range(len(eig_vals))]
    eig_pairs = sorted(eig_pairs,key=lambda k: k[0], reverse=True)

    #W = np.array([eig_pairs[i][1] for i in range(num_eigvec)])
    W = np.hstack([eig_pairs[i][1].reshape(array_len,1) for i in range(num_eigvec)])
    
    return(W)

## Function : Error Matrix
def Err_Mat(key_df,mat_array): 
    mat_err = np.zeros((mat_array.shape[0],mat_array.shape[0]))
    dict_err = {}

    for i in range(mat_array.shape[0]):
        for j in range(mat_array.shape[0]):
            # L2 Norm
            dist_err = np.linalg.norm(mat_array[i,:] - mat_array[j,:],2)
            mat_err[i,j] = dist_err
            if i!=j :
                key_err = ((key_df.iloc[i],key_df.iloc[j]),(i,j))
                dict_err[key_err] = dist_err

    return(mat_err,dict_err)

## Function : Distribution Curve
def Dist_Curv(tf_list,err_dict):

    print("Distribution Curve")

    dist_true = [list(err_dict.values())[i] for i in range(len(err_dict)) if tf_list[i]==True]
    dist_false = [list(err_dict.values())[i] for i in range(len(err_dict)) if tf_list[i]==False]

    plt.hist(dist_true,bins='auto',normed=1,histtype='step',color='blue',label='Dist_True')
    plt.hist(dist_false,bins='auto',normed=1,histtype='step',color='red',label='Dist_False')
    plt.legend(loc='upper right')
    plt.title("Distribution Curve")

    plt.show()

    return(dist_true,dist_false)

## Function : EER Curve
def EER_Curve(tf_list,err_dict,sampling_thres):

    print("EER Curve")

    eer_df = pd.DataFrame(columns = ['thres','fn','fp'])
    err_values = err_dict.values()
    n_thres = int(sampling_thres*len(err_values))
    sampled_err_values = random.sample(list(err_values),n_thres)

    for i, thres in enumerate(set(sampled_err_values)):
        predicted_tf = [e <= thres for e in err_dict.values()]
        
        tn, fp, fn, tp = confusion_matrix(tf_list,predicted_tf).ravel()

        eer_ser = {'thres':thres,'tn':tn,'fp':fp,'fn':fn,'tp':tp}
        eer_df = eer_df.append(eer_ser,ignore_index=True)
        
        curr_percent = 100 * (i+1) / n_thres
        if (curr_percent % 10)==0 : print(int(curr_percent),end="|")

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
    plt.text(max(eer_df_graph.thres)*0.9,min_te_val-10,'EER : ' + str(round(min_te_val,2)))
    plt.legend()
    plt.title("EER Curve")

    plt.show()

    return(eer_df)



# Main

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
import time

# import ORLDB images
# image size 56 x 46

start_time = time.time()
array_len = 56*46

# For Windows
file_path = 'D:\Matlab_Drive\Data\ORLDB'

# For Linux
# file_path = '/home/hero/Matlab_Drive/Data/ORLDB'

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

# Apply LDA

w_train = LDA_ORLDB(image_train,array_len,len(image_raw.person.unique())-1)

mat_train = [np.array(img.reshape(1,array_len)) for img in image_train.image]
mat_train = np.vstack(mat_train)
mat_test = [np.array(img.reshape(1,array_len)) for img in image_test.image]
mat_test = np.vstack(mat_test)

lda_train = mat_train.dot(w_train)
lda_test = mat_test.dot(w_train)

## Error Matrix
mat_err,dict_err = Err_Mat(image_test.person,lda_test)

# Distribution Curve
person_comp = [(person[0],person[1]) for person,num in list(dict_err.keys())]
person_tf = [person[0]==person[1] for person in person_comp]
dist_true,dist_false = Dist_Curv(person_tf,dict_err)

## EER Curve
eer_df = EER_Curve(person_tf,dict_err,0.1)

## Accuracy
pred_list = []
for i in range(lda_test.shape[0]):
    dist_list = []
    for j in range(lda_train.shape[0]):
        # L2 Norm
        dist_err = np.linalg.norm(lda_test[i,:] - lda_train[j,:],2)
        dist_list.append(dist_err)

    pred_person = image_train.person.iloc[np.argmin(dist_list)]
    pred_list.append(pred_person)


actual_list = list(image_train.person.values)
acc_tflist = [pred_list[i]==actual_list[i] for i in range(len(pred_list))]
acc = sum(acc_tflist) / len(acc_tflist)

print("Accuracy : " + str(acc) + " [" + str(sum(acc_tflist)) + "/" + str(len(acc_tflist)) + "]")

print("Elapsed Time : " + str(time.time() - start_time))

'''