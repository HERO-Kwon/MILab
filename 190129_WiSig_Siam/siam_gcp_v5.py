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
import numpy.random as rng
from keras.layers import Input, Conv2D, Lambda, merge, Dense, Flatten,MaxPooling2D
from keras.models import Model, Sequential
from keras.regularizers import l2
from keras import backend as K
from keras.optimizers import SGD,Adam
from keras.losses import binary_crossentropy

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

# Siamese Net
def W_init(shape,name=None):
    """Initialize weights as in paper"""
    values = rng.normal(loc=0,scale=1e-2,size=shape)
    return K.variable(values,name=name)
#//TODO: figure out how to initialize layer biases in keras.
def b_init(shape,name=None):
    """Initialize bias as in paper"""
    values=rng.normal(loc=0.5,scale=1e-2,size=shape)
    return K.variable(values,name=name)

# Network
input_shape = (500, 30, 6)
left_input = Input(input_shape)
right_input = Input(input_shape)
#build convnet to use in each siamese 'leg'
convnet = Sequential()
convnet.add(Conv2D(64,(3,3),activation='relu',input_shape=input_shape,
                   kernel_initializer=W_init,kernel_regularizer=l2(2e-4)))
convnet.add(MaxPooling2D())
convnet.add(Conv2D(128,(3,3),activation='relu',
                   kernel_regularizer=l2(2e-4),kernel_initializer=W_init,bias_initializer=b_init))
convnet.add(MaxPooling2D())
convnet.add(Conv2D(256,(3,3),activation='relu',
                   kernel_regularizer=l2(2e-4),kernel_initializer=W_init,bias_initializer=b_init))
#convnet.add(MaxPooling2D())
#convnet.add(MaxPooling2D(2,2))
convnet.add(Flatten())
convnet.add(Dense(1024,activation="sigmoid",kernel_regularizer=l2(1e-3),kernel_initializer=W_init,bias_initializer=b_init))


#call the convnet Sequential model on each of the input tensors so params will be shared
encoded_l = convnet(left_input)
encoded_r = convnet(right_input)
#layer to merge two encoded inputs with the l1 distance between them
L1_layer = Lambda(lambda tensors:K.abs(tensors[0] - tensors[1]))
#call this layer on list of two input tensors.
L1_distance = L1_layer([encoded_l, encoded_r])
prediction = Dense(1,activation='sigmoid',bias_initializer=b_init)(L1_distance)
siamese_net = Model(inputs=[left_input,right_input],outputs=prediction)

optimizer = Adam(0.0001)#Adam(0.00006)
#//TODO: get layerwise learning rates and momentum annealing scheme described in paperworking
siamese_net.compile(loss="binary_crossentropy",optimizer=optimizer)

siamese_net.count_params()

class Siamese_Loader:
    """For loading batches and testing tasks to a siamese net"""
    def __init__(self, data_list, data_subsets = ["train", "val"]):
        self.data = {}
        self.categories = {}
        self.info = {}
        
        X,Xval,Y,Yval = data_list
        Ys = np.concatenate([Y,Yval],axis=0)
        
        # make category
        c = {}
        cval = {}
        lab_ser = pd.Series(Ys[:,0].astype('int'))
        lab_ser_tr = lab_ser.loc[train_index].reset_index()
        lab_ser_te = lab_ser.loc[test_index].reset_index()
                    
        for num in np.unique(lab_ser_tr[0]):
            c[num] = list(lab_ser_tr[lab_ser_tr[0]==num].index)
            for num in np.unique(lab_ser_te[0]):
                cval[num] = list(lab_ser_te[lab_ser_te[0]==num].index)

        self.data["train"] = X
        self.data["val"] = Xval
        self.categories["train"] = c
        self.categories["val"] = cval
        
    def get_batch(self,batch_size,s="train"):
        """Create batch of n pairs, half same class, half different class"""
        X=self.data[s]
        c=self.categories[s]
        n_classes, n_examples, w, h, ch = X.shape

        #randomly sample several classes to use in the batch
        categories = rng.choice(n_classes,size=(batch_size,),replace=False)
        #initialize 2 empty arrays for the input image batch
        pairs=[np.zeros((batch_size, w,h,ch)) for i in range(2)]
        #initialize vector for the targets, and make one half of it '1's, so 2nd half of batch has same class
        targets=np.zeros((batch_size,))
        targets[batch_size//2:] = 1
        for i in range(batch_size):
            category = categories[i]
            idx_1 = rng.randint(0, n_examples)
            pairs[0][i,:,:,:] = X[category, idx_1].reshape(w, h, ch)
            idx_2 = rng.randint(0, n_examples)
            #pick images of same class for 1st half, different for 2nd
            cat_key = get_key(category,c)
            cat_same = c[cat_key]
            cat_diff = list(set(range(n_classes)) - set(cat_same))
            if i >= batch_size // 2:
                #category_2 = category  
                category_2 = rng.choice(cat_same,size=1,replace=False)[0]
            else: 
                #add a random number to the category modulo n classes to ensure 2nd image has
                # ..different category
                #category_2 = (category + rng.randint(1,n_classes)) % n_classes
                category_2 = rng.choice(cat_diff,size=1,replace=False)[0]
                
            pairs[1][i,:,:,:] = X[category_2,idx_2].reshape(w, h,ch)
        return pairs, targets
    
    def generate(self, batch_size, s="train"):
        """a generator for batches, so model.fit_generator can be used. """
        while True:
            pairs, targets = self.get_batch(batch_size,s)
            yield (pairs, targets)    
            
    def get_score(self,model,pairs):
        probs = model.predict(pairs)
        return probs
    
    def val_dist(self,model):
        Xval = self.data["val"]
        cval = self.categories["val"]
        
        n_classes, n_examples, h, w, ch = Xval.shape
        res_list = []
        for i in range(n_classes):
            test_image = []
            for j in range(n_classes):
                test_image.append(Xval[i,0])
            test_images = np.array(test_image)

            support_images = Xval[:,0]
            pairs = [test_images,support_images]
            val_score = self.get_score(model,pairs)

            score_list = [[i,ti,int(get_key(i,cval) == get_key(ti,cval)),val_score[0][0]] for ti in range(n_classes)]
            [res_list.append(score_list[ii]) for ii in range(len(score_list))]

        res_df = pd.DataFrame(res_list,columns=['i','ti','truth','score'])
        res_df1 = res_df[res_df.i != res_df.ti]
        return(res_df1)
    
    def eer_graphs(self,dist_truth,dist_score,pos_label):
        fpr, tpr, thresholds = roc_curve(dist_truth, dist_score,pos_label=pos_label)
        eer = brentq(lambda x : 1. - x - interp1d(fpr, tpr)(x), 0., 1.)
        
        # ROC
        plt.plot(fpr, tpr, '.-')#, label=self.model_name)
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
    
    def train(self, model, epochs, verbosity):
        model.fit_generator(self.generate(batch_size),
    
                             )



# Main
list_cv = [2]

prep_method = 0 # 0:none,1:pca_HC
locs = [1]
dirs = [1]
use_ratio = 0.5

#Training var
loss_every=10 # interval for printing loss (iterations)
batch_size = 32
n_iter = 5000

## Data
PATH = "/home/herokwon/mount_data/Data/Wi-Fi_HC/180_100/"
#PATH = "D:\\Matlab_Drive\\Data\\WIFI\\180_100\\"

Xs,Ys,resXs,resYs = ReadCSIData(PATH,use_ratio,locs,dirs)
prep_xs = np.mean(np.squeeze(Xs),axis=2).reshape([-1,500*6])
prep_ys = np.squeeze(Ys)

for n_splits in list_cv:
    skf = StratifiedKFold(n_splits,random_state=10)
    for i,[train_index,test_index] in enumerate(skf.split(prep_xs,prep_ys)):
            print('Splits:' + str(i))

            X = Xs[train_index]
            Xval = Xs[test_index]
            Y = Ys[train_index]
            Yval = Ys[test_index]
                    
            loader = Siamese_Loader([X,Xval,Y,Yval])

            print("training")
            t1 = time.time()            
            for i in range(1, n_iter+1):
                
                (inputs,targets)=loader.get_batch(batch_size)
                loss=siamese_net.train_on_batch(inputs,targets)
                
                if i % loss_every == 0:
                    print("iteration {}, training loss: {:.2f},".format(i,loss))

            print("Time:" + str(time.time() - t1))

            # validate

            loss=siamese_net.train_on_batch(inputs,targets)
            if i % loss_every == 0:
                print("iteration {}, training loss: {:.2f},".format(i,loss))
            
            dist_lsep = loader.val_dist(siamese_net)
            eer_siam = loader.eer_graphs(dist_lsep.truth,dist_lsep.score,0)

            