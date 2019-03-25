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

## Functions
# function to return key for any value 
def get_key(val,my_dict): 
    for key, list_val in my_dict.items(): 
         if val in list_val: 
             return key 
  
    return "key doesn't exist"

def ReadCSIData(PATH,ratio,locs,dirs):
    print("loading data from {}".format(PATH))
    list_csi = []
    list_lab = []
    list_res_csi = []
    list_res_lab = []
    
    ids = int(ratio * 100)
    res_ids = int((1-ratio)*100)

    # Filter Loc,Dir
    file_list = []
    for i in locs:
        for j in dirs:
            filename = 'Dataset_' + str(i) + '_' + str(j) + '.npy'
            file_list.append(filename)

    for file in file_list:
        data_read = np.load(PATH + file)
        csi_read = data_read[:,4:].astype('float32')
        lab_read = data_read[:,0].astype('int')

        data_x = csi_read.reshape([-1,10,6,30,500]).swapaxes(2,4)

        uniq_label = np.unique(lab_read)
        label_table = pd.Series(np.arange(len(uniq_label)),index=uniq_label)
        data_y = np.array([label_table[num] for num in lab_read]).reshape([-1,10])
        
        # use half of the dataset
        idx_ids = (data_y[:,0] < ids)
        list_csi.append(data_x[idx_ids])
        list_lab.append(data_y[idx_ids])
        
        reserve_ids = (data_y[:,0] >= ids)
        list_res_csi.append(data_x[reserve_ids])
        list_res_lab.append(data_y[reserve_ids])

    arr_csi = np.array(list_csi).swapaxes(0,1).reshape([ids*len(file_list),10,500,30,6])
    arr_lab = np.array(list_lab).swapaxes(0,1).reshape([ids*len(file_list),10])
    Xs = arr_csi.reshape([-1,1,500,30,6])
    Ys = arr_lab.reshape([-1,1])

    reserve_csi = np.array(list_res_csi).swapaxes(0,1).reshape([res_ids*len(file_list),10,500,30,6])
    reserve_lab = np.array(list_res_lab).swapaxes(0,1).reshape([res_ids*len(file_list),10])
    reserve_Xs = reserve_csi.reshape([-1,1,500,30,6])
    reserve_Ys = reserve_lab.reshape([-1,1])

    return([Xs,Ys,reserve_Xs,reserve_Ys])

class LinearModels:
    isdual = 1 #prim:0,dual:1
    model_name = 'Default'
    train_time = 0

    def __init__(self,data):
        self.X = data[0]
        self.Y = data[1]
        self.Xval = data[2]
        self.Yval = data[3]
    
    def model(self):
        pass
    
    def train(self):
        t1 = time.time()
        alpha_mat = self.model()
        LinearModels.train_time = time.time()-t1
        print("Training Time: " + str(LinearModels.train_time))
        return(alpha_mat)

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

## Main

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


prep_method = 1 # 0:none,1:pca_HC
locs = [1]
dirs = [1]
use_ratio = 0.7
## Data
PATH = "/home/herokwon/mount_data/Data/Wi-Fi_HC/180_100/"
#PATH = "D:\\Matlab_Drive\\Data\\WIFI\\180_100\\"

Xs,Ys,resXs,resYs = ReadCSIData(PATH,use_ratio,locs,dirs)
prep_xs = np.mean(np.squeeze(Xs),axis=2).reshape([-1,500*6])
prep_ys = np.squeeze(Ys)

m_results = pd.DataFrame(columns = ['name','splits','splits_number','random_state','eer','ter_num','tr_time'])
list_cv = [2,5]
list_rs = [10,20,30]


for n_splits in list_cv:
    print('CV:' + str(n_splits))
    for rs_num in list_rs:
        print('RandomState:' + str(rs_num))

        skf = StratifiedKFold(n_splits,random_state=rs_num)

        for i,[train_index,test_index] in enumerate(skf.split(prep_xs,prep_ys)):
            print('Splits:' + str(i))

            X = prep_xs[train_index]
            Xval = prep_xs[test_index]
            Y = prep_ys[train_index]
            Yval = prep_ys[test_index]

            if prep_method==1:
                pca_c = len(np.unique(Y))
                pca_s = int(Y.shape[0] / pca_c)

                pca_v,pca_u = PCA_HC(X,pca_c,pca_s,40)
                X = np.dot(pca_v,X.T-pca_u).T
                Xval = np.dot(pca_v,Xval.T-pca_u).T


            # LSE Prim
            m_lsep = LSEprim([X,Y,Xval,Yval])
            dist_lsep = m_lsep.val_dist(m_lsep.train()) 
            eer_lsep = m_lsep.eer_graphs(dist_lsep.truth,dist_lsep.score,0)

            #gather results
            res_ser = pd.Series({'name':m_lsep.model_name,'splits':n_splits,'splits_number':i,
                                'random_state':rs_num,'eer':eer_lsep,'ter_num':0,'tr_time':m_lsep.train_time}) 
            m_results = m_results.append(res_ser,ignore_index=True)            
            
            # LSE Dual
            m_lsed = LSEdual([X,Y,Xval,Yval])
            dist_lsed = m_lsed.val_dist(m_lsed.train()) 
            eer_lsed = m_lsed.eer_graphs(dist_lsed.truth,dist_lsed.score,0)

            #gather results
            res_ser = pd.Series({'name':m_lsed.model_name,'splits':n_splits,'splits_number':i,
                                'random_state':rs_num,'eer':eer_lsed,'ter_num':0,'tr_time':m_lsed.train_time}) 
            m_results = m_results.append(res_ser,ignore_index=True)
            
            for m in range(4):
                print('TER_m:' + str(m))
                
                # TER Prim
                m_terp = TERprim([X,Y,Xval,Yval],0.5,0.5,m)
                dist_terp = m_terp.val_dist(m_terp.train()) 
                eer_terp = m_terp.eer_graphs(dist_terp.truth,dist_terp.score,0)

                #gather results
                res_ser = pd.Series({'name':m_terp.model_name,'splits':n_splits,'splits_number':i,
                                    'random_state':rs_num,'eer':eer_terp,'ter_num':m,'tr_time':m_terp.train_time}) 
                m_results = m_results.append(res_ser,ignore_index=True)
                
                # TER Dual
                m_terd = TERdual([X,Y,Xval,Yval],0.5,0.5,m)
                dist_terd = m_terd.val_dist(m_terd.train()) 
                eer_terd = m_terd.eer_graphs(dist_terd.truth,dist_terd.score,0)

                #gather results
                res_ser = pd.Series({'name':m_terd.model_name,'splits':n_splits,'splits_number':i,
                                    'random_state':rs_num,'eer':eer_terd,'ter_num':m,'tr_time':m_terd.train_time}) 
                m_results = m_results.append(res_ser,ignore_index=True)
        m_results.to_csv('m_results_pca.csv')