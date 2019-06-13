import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedKFold
import time
import random
from sklearn.metrics import confusion_matrix
from keras.layers import Input, Conv2D, Lambda, merge, Dense, Flatten,MaxPooling2D
from keras.models import Model, Sequential
from keras.regularizers import l2
from keras import backend as K
from keras.optimizers import SGD,Adam
from keras.losses import binary_crossentropy
import numpy.random as rng
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.utils import shuffle
from scipy.optimize import brentq
from scipy.interpolate import interp1d
from sklearn.metrics import roc_curve
import pickle
import gzip

%matplotlib inline

#Data load
path_fc = 'D:\\Data\\WIFI\\wisig_kar\\'
with gzip.open(path_fc+'arr_fc1'+'.pickle.gz','rb') as f:
    arr_fc1 = pickle.load(f)
with gzip.open(path_fc+'arr_fc2'+'.pickle.gz','rb') as f:
    arr_fc2 = pickle.load(f)
with gzip.open(path_fc+'arr_res'+'.pickle.gz','rb') as f:
    arr_score = pickle.load(f)


import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time
import random
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import confusion_matrix
from sklearn.utils import shuffle
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.optimize import brentq
from scipy.interpolate import interp1d
from sklearn.metrics import roc_curve
from sklearn.svm import SVC

## Functions
# function to return key for any value 
def get_key(val,my_dict): 
    for key, list_val in my_dict.items(): 
         if val in list_val: 
             return key 
  
    return "key doesn't exist"

class LinearModels:
    isdual = 0 #prim:0,dual:1
    model_name = 'Default'
    train_time = 0

    def __init__(self,data):
        self.X = data[0]
        self.Y = data[1]
        self.Xval = data[2]
        self.Yval = data[3]
    
    def model(self):
        if self.isdual:
            return(np.eye(X.shape[0],X.shape[0]))
        else:
            return(np.eye(X.shape[1],X.shape[1]))

    def train(self):
        t1 = time.time()
        alpha_mat = self.model()
        LinearModels.train_time = time.time()-t1
        print("Training Time: " + str(LinearModels.train_time))
        return(alpha_mat)
    def predict(self,alpha_mat):
        if self.isdual:
            res_mat = np.dot(self.Xval,self.X.T).dot(alpha_mat)
        else:
            res_mat = self.Xval.dot(alpha_mat)
        return(res_mat)
    def val_dist(self,alpha_mat):
        if self.isdual:
            val_mat = np.dot(self.Xval,self.X.T).dot(alpha_mat)
        else:
            val_mat = self.Xval.dot(alpha_mat)

        res_list = []
        for i in range(val_mat.shape[0]):
            for ti in range(val_mat.shape[0]):
                val_dist = val_mat[i,:] - val_mat[ti,:]
                val_score = np.sqrt(np.sum(np.multiply(val_dist,val_dist)))
                grd_truth =  int(self.Yval[i] == self.Yval[ti])
                score_list = [i,ti,grd_truth,val_score]
                res_list.append(score_list)
        
        res_df = pd.DataFrame(res_list,columns=['i','ti','truth','score'])
        return(res_df[res_df.i != res_df.ti])
    
    def eer_graphs(self,dist_truth,dist_score,pos_label):
        fpr, tpr, thresholds = roc_curve(dist_truth, dist_score,pos_label=pos_label)
        eer = brentq(lambda x : 1. - x - interp1d(fpr, tpr)(x), 0., 1.)
        
        # ROC
        plt.plot(fpr, tpr, '.-', label=self.model_name)
        plt.plot([0, 1], [0, 1], 'k--', label="random guess")
        
        plt.xlabel('False Positive Rate (Fall-Out)')
        plt.ylabel('True Positive Rate (Recall)')
        plt.title('Receiver Operating Characteristic curve')
        plt.legend()
        plt.show()
        
        # EER
        thres_mask = thresholds <= 5*np.median(dist_score)
        
        th_mask = thresholds[thres_mask]
        fpr_mask= fpr[thres_mask]
        tpr_mask = tpr[thres_mask]
        
        plt.plot(th_mask,fpr_mask,color='blue', label="FPR")
        plt.plot(th_mask,1-tpr_mask,color='red',label="FNR")
        plt.plot(th_mask,fpr_mask + (1-tpr_mask),color='green',label="TER")
        plt.axhline(eer,color='black')
        plt.text(max(th_mask)*1.05,eer,'EER : ' + str(round(eer*100,1)))

        plt.xlabel('Thresholds')
        plt.ylabel('Error Rates (%)')
        plt.title('Equal Error Rate')
        plt.legend()
        plt.show()
        
        return(eer)

class LSEprim(LinearModels):
    model_name = 'LSE:Prim'
    isdual = 0
    def model(self):
        alpha_lse = []
        for k in list(set(self.Y)):
            P_lse = self.X
            I_lse = np.eye(P_lse.shape[1])
            b_lse = 10**(-4)
            y_lse = (self.Y==k).astype('int')

            ak_lse = np.dot(np.dot(np.linalg.pinv(b_lse*I_lse + P_lse.T.dot(P_lse)),P_lse.T),y_lse)
            alpha_lse.append(ak_lse) 
        
        return(np.array(alpha_lse).T)

class LSEdual(LinearModels):
    model_name = 'LSE:Dual'
    isdual = 1
    def model(self):
        beta_lsed = []
        for k in list(set(self.Y)):
            P_lsed = self.X
            I_lsed = np.eye(P_lsed.shape[0])
            b_lsed = 10**(-4)
            y_lsed = (self.Y==k).astype('int')

            bk_lsed = np.dot(np.linalg.pinv(np.dot(P_lsed,P_lsed.T) + b_lsed*I_lsed),y_lsed)
            beta_lsed.append(bk_lsed)

        return(np.array(beta_lsed).T)

class TERprim(LinearModels):
    model_name = 'TER:Prim'
    isdual = 0
        
    M_dict = {}
    M_dict['0'] = (0,1),(0,1)
    M_dict['1'] = (1,0),(1,0)
    M_dict['2'] = (1,0),(0,1)
    M_dict['3'] = (0,1),(1,0)
    M_dict['4'] = (1,0),(0.5,0)
    M_dict['5'] = (0.5,0),(1,0)
    M_dict['6'] = (1,0),(0.7,0)
    M_dict['7'] = (0.7,0),(1,0)
    M_dict['8'] = (1,0),(0.3,0)
    M_dict['9'] = (0.3,0),(1,0)
    
    def __init__(self,data,r,n,m_pos):
        super(TERprim, self).__init__(data)
        self.r = r
        self.n = n
        self.mod_w = TERprim.M_dict[str(m_pos)]
    def model(self):
        #simplified version of TER
        alpha = []
        for k in list(set(self.Y)):
            P = self.X
            b_ter = 10**(-4)
            I_ter = np.eye(P.shape[1])

            mk_n = self.X[self.Y!=k].shape[0]
            mk_p = self.X[self.Y==k].shape[0]

            w_n = self.mod_w[0][0] * 1/mk_n + self.mod_w[0][1]
            w_p = self.mod_w[1][0] * 1/mk_p + self.mod_w[1][1]

            ones_mkn = 1*(self.Y!=k) *w_n
            ones_mkp = 1*(self.Y==k) *w_p

            W = np.zeros((len(self.Y), len(self.Y)), float)
            np.fill_diagonal(W,ones_mkn+ones_mkp)
            yk = ((self.r-self.n)*ones_mkn+(self.r+self.n)*ones_mkp).T
            ak = np.linalg.pinv(b_ter*I_ter + (P.T).dot(W).dot(P)).dot(P.T).dot(W).dot(yk)

            alpha.append(ak)
        return(np.array(alpha).T)

class TERdual(LinearModels):
    model_name = 'TER:Dual'
    isdual = 1
    
    M_dict = {}
    M_dict['0'] = (0,1),(0,1)
    M_dict['1'] = (1,0),(1,0)
    M_dict['2'] = (1,0),(0,1)
    M_dict['3'] = (0,1),(1,0)
    M_dict['4'] = (1,0),(0.5,0)
    M_dict['5'] = (0.5,0),(1,0)
    M_dict['6'] = (1,0),(0.7,0)
    M_dict['7'] = (0.7,0),(1,0)
    M_dict['8'] = (1,0),(0.3,0)
    M_dict['9'] = (0.3,0),(1,0)
    
    def __init__(self,data,r,n,m_pos):
        super(TERdual, self).__init__(data)
        self.r = r
        self.n = n
        self.mod_w = TERdual.M_dict[str(m_pos)]
    def model(self):
        #TER
        beta = []
        for k in list(set(self.Y)):
            P = self.X
            b_ter = 10**(-4)
            I_ter = np.eye(P.shape[0])

            mk_n = self.X[self.Y!=k].shape[0]
            mk_p = self.X[self.Y==k].shape[0]

            w_n = self.mod_w[0][0] * 1/mk_n + self.mod_w[0][1]
            w_p = self.mod_w[1][0] * 1/mk_p + self.mod_w[1][1]

            ones_mkn = 1*(self.Y!=k) *w_n
            ones_mkp = 1*(self.Y==k) *w_p

            W = np.zeros((len(self.Y), len(self.Y)), float)
            np.fill_diagonal(W,ones_mkn+ones_mkp)
            yk = ((self.r-self.n)*ones_mkn+(self.r+self.n)*ones_mkp).T
            
            bk = np.linalg.pinv(P.dot(P.T).dot(W)+b_ter*I_ter).dot(yk)
            
            beta.append(bk)
        return(np.array(beta).T)

class SVMlinear(LinearModels):
    model_name = 'SVM:Linear'
    isdual = 0 

    def train(self,c):
        t1 = time.time()
        clf = SVC(kernel='linear',C=c,probability=True)
        clf.fit(self.X,self.Y)
        prob_mat = clf.predict_proba(self.Xval)
        
        LinearModels.train_time = time.time()-t1
        print("Training Time: " + str(LinearModels.train_time))
        return(prob_mat)
    
    def val_dist(self,val_mat):
        res_list = []
        for i in range(val_mat.shape[0]):
            for ti in range(val_mat.shape[0]):
                val_dist = val_mat[i,:] - val_mat[ti,:]
                val_score = np.sqrt(np.sum(np.multiply(val_dist,val_dist)))
                grd_truth =  int(self.Yval[i] == self.Yval[ti])
                score_list = [i,ti,grd_truth,val_score]
                res_list.append(score_list)
        
        res_df = pd.DataFrame(res_list,columns=['i','ti','truth','score'])
        return(res_df[res_df.i != res_df.ti])

class SVMrbf(LinearModels):
    model_name = 'SVM:RBF'
    isdual = 0 
            
    def train(self,c,gamma):
        t1 = time.time()
        clf = SVC(kernel='rbf',C=c,gamma=gamma,probability=True)
        clf.fit(self.X,self.Y)
        prob_mat = clf.predict_proba(self.Xval)
        
        LinearModels.train_time = time.time()-t1
        print("Training Time: " + str(LinearModels.train_time))
        return(prob_mat)
    
    def val_dist(self,val_mat):
        res_list = []
        for i in range(val_mat.shape[0]):
            for ti in range(val_mat.shape[0]):
                val_dist = val_mat[i,:] - val_mat[ti,:]
                val_score = np.sqrt(np.sum(np.multiply(val_dist,val_dist)))
                grd_truth =  int(self.Yval[i] == self.Yval[ti])
                score_list = [i,ti,grd_truth,val_score]
                res_list.append(score_list)
        
        res_df = pd.DataFrame(res_list,columns=['i','ti','truth','score'])
        return(res_df[res_df.i != res_df.ti])

class SVMpoly(LinearModels):
    model_name = 'SVM:Poly'
    isdual = 0 
            
    def train(self,c,degree):
        t1 = time.time()
        clf = SVC(kernel='poly',C=c,degree=degree,probability=True)
        clf.fit(self.X,self.Y)
        prob_mat = clf.predict_proba(self.Xval)
        
        LinearModels.train_time = time.time()-t1
        print("Training Time: " + str(LinearModels.train_time))
        return(prob_mat)
    
    def val_dist(self,val_mat):
        res_list = []
        for i in range(val_mat.shape[0]):
            for ti in range(val_mat.shape[0]):
                val_dist = val_mat[i,:] - val_mat[ti,:]
                val_score = np.sqrt(np.sum(np.multiply(val_dist,val_dist)))
                grd_truth =  int(self.Yval[i] == self.Yval[ti])
                score_list = [i,ti,grd_truth,val_score]
                res_list.append(score_list)
        
        res_df = pd.DataFrame(res_list,columns=['i','ti','truth','score'])
        return(res_df[res_df.i != res_df.ti])
# Prep Method
def PCA_HC(prep_xs,c,s,num_eig):
    M = c*s
    X = prep_xs.T
    ones_vec = np.ones([M,1])
    u = (1/M * X).dot(ones_vec)
    A = X - u.dot(ones_vec.T)
    Z = A.dot(A.T)
    W,V= np.linalg.eigh(Z)
    return(V[:,-num_eig:].T,u)

def RMmodel(X,order):
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

fc1 = arr_fc1.squeeze()
fc1lab= arr_score[:,0]
fc2 = arr_fc2.squeeze()
fc2lab = arr_score[:,1]

X = fc1
Y = fc1lab
Xval = fc2
Yval = fc2lab

m_results = pd.DataFrame(columns = ['name','splits','splits_number','random_state','eer','param0','tr_time'])
#list_cv = [2]
list_rs = [10]

n_splits = 0
i=0

for rs_num in list_rs:
    print('RandomState:' + str(rs_num))


    #Defalut HC's
    m_hc = LinearModels([X,Y,Xval,Yval])
    m_hc.isdual=0
    #dist_hc = m_hc.val_dist(m_hc.train())
    #eer_hc = m_hc.eer_graphs(dist_hc.truth,dist_hc.score,0)
    #gather results
    #res_ser = pd.Series({'name':m_hc.model_name,'splits':n_splits,'splits_number':i,
    #                    'random_state':rs_num,'eer':eer_hc,'param0':0,'tr_time':m_hc.train_time}) 
    #m_results = m_results.append(res_ser,ignore_index=True)  

    # LSE Prim
    m_lsep = LSEprim([X,Y,Xval,Yval])
    # LSE Dual
    m_lsed = LSEdual([X,Y,Xval,Yval])
    #dist_lsed = m_lsed.val_dist(m_lsed.train()) 
    #eer_lsed = m_lsed.eer_graphs(dist_lsed.truth,dist_lsed.score,0)

    #gather results
    #res_ser = pd.Series({'name':m_lsed.model_name,'splits':n_splits,'splits_number':i,
    #                    'random_state':rs_num,'eer':eer_lsed,'param0':0,'tr_time':m_lsed.train_time}) 
    #m_results = m_results.append(res_ser,ignore_index=True)


import time
t1 = time.time()
#LSE output
lse_a = m_lsep.predict(m_lsep.train())
print(time.time()-t1)

lse_aa = np.argmax(lse_a,axis=1)

lse_tf = [int(lse_aa[i]==Yval[i]) for i in range(len(Yval))]
np.sum(lse_tf) / len(lse_tf)

