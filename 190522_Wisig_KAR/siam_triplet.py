import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedKFold
import time
import random
from sklearn.metrics import confusion_matrix
from keras.layers import Input, Conv2D, Lambda, merge, Dense, Flatten,MaxPooling2D, concatenate
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
import tensorflow as tf

%matplotlib inline

def triplet_loss(y_true, y_pred):
    fc_size = 1024
    alpha = 0.01
    a_pred = y_pred[:, 0:fc_size]
    p_pred = y_pred[:, fc_size:2*fc_size]
    n_pred = y_pred[:, 2*fc_size:3*fc_size]
    pos_dist = tf.reduce_sum(tf.square(tf.subtract(a_pred, p_pred)), axis=-1)
    neg_dist = tf.reduce_sum(tf.square(tf.subtract(a_pred, n_pred)), axis=-1)
    basic_loss = tf.add(tf.subtract(pos_dist, neg_dist), alpha)
    loss = tf.reduce_sum(tf.maximum(basic_loss, 0.0))
    return loss

# Network
input_shape = (500, 30, 6)
ank_input = Input(input_shape)
pos_input = Input(input_shape)
neg_input = Input(input_shape)
#build convnet to use in each siamese 'leg'
convnet = Sequential()
convnet.add(Conv2D(64,(3,3),activation='relu',input_shape=input_shape, padding='same'))
                   #kernel_initializer=W_init,kernel_regularizer=l2(1e-4)))
convnet.add(MaxPooling2D())
convnet.add(Conv2D(128,(3,3),activation='relu', padding='same'))
                   #kernel_regularizer=l2(1e-4),kernel_initializer=W_init,bias_initializer=b_init))
convnet.add(MaxPooling2D())
convnet.add(Conv2D(256,(3,3),activation='relu', padding='same'))
                   #kernel_regularizer=l2(1e-4),kernel_initializer=W_init,bias_initializer=b_init))
convnet.add(MaxPooling2D())
convnet.add(Conv2D(512,(3,3),activation='relu', padding='same'))
#convnet.add(MaxPooling2D(2,2))
convnet.add(Flatten())
#convnet.add(Dense(1024,activation="sigmoid"))#,kernel_initializer=W_init,bias_initializer=b_init))
convnet.add(Dense(1024))#,activation="sigmoid"))#,kernel_initializer=W_init,bias_initializer=b_init))


#call the convnet Sequential model on each of the input tensors so params will be shared
encoded_a = convnet(ank_input)
encoded_p = convnet(pos_input)
encoded_n = convnet(neg_input)


normalize = Lambda(lambda x: K.l2_normalize(x, axis=-1), name='normalize')
output_a = normalize(encoded_a)
output_p = normalize(encoded_p)
output_n = normalize(encoded_n)

#merged_vector = concatenate([encoded_a, encoded_p, encoded_n], axis=-1)
merged_vector = concatenate([output_a, output_p, output_n], axis=-1)
siamese_net = Model(inputs=[ank_input, pos_input, neg_input],outputs=merged_vector)


optimizer = Adam(0.1)#Adam(0.00006)
sgd = SGD(lr=1e-5, momentum=0.9, nesterov=True, decay=1e-6)
#//TODO: get layerwise learning rates and momentum annealing scheme described in paperworking
siamese_net.compile(loss=triplet_loss,optimizer=optimizer)

#siamese_net.count_params()
siamese_net.summary()


# function to return key for any value 
def get_key(val,my_dict): 
    for key, list_val in my_dict.items(): 
         if val in list_val: 
             return key 
  
    return "key doesn't exist"


def ReadData(PATH,n_splits):
    print("loading data from {}".format(PATH))
    list_csi = []
    list_lab = []

    locs = [1]
    dirs = [1,2,3,4]

    # Filter Loc,Dir
    file_list = []
    for i in locs:
        for j in dirs:
            filename = 'Dataset_' + str(i) + '_' + str(j) + '.npy'
            file_list.append(filename)
    # Filter Dir
    
    for file in file_list:
        data_read = np.load(PATH + file)
        csi_read = data_read[:,4:].astype('float32')
        lab_read = data_read[:,0].astype('int')

        data_x = csi_read.reshape([-1,10,6,30,500]).swapaxes(2,4)

        uniq_label = np.unique(lab_read)
        label_table = pd.Series(np.arange(len(uniq_label)),index=uniq_label)
        data_y = np.array([label_table[num] for num in lab_read]).reshape([-1,10])
        
        # use half of the dataset
        idx_half = (data_y[:,0] < 50)
        
        list_csi.append(data_x[idx_half])
        list_lab.append(data_y[idx_half])
        
    arr_csi = np.array(list_csi).swapaxes(0,1).reshape([50*len(file_list),10,500,30,6])
    arr_lab = np.array(list_lab).swapaxes(0,1).reshape([50*len(file_list),10])

    skf = StratifiedKFold(n_splits,random_state=10)
    data_list = []

    arr_csi1 = arr_csi.reshape([-1,500,30,6])
    arr_lab1 = arr_lab.reshape([-1,1])

    for train_index,test_index in skf.split(arr_csi1,arr_lab1):
        
    #idx_tr,idx_te = train_test_split(np.arange(len(arr_csi)),test_size=0.2,random_state=10)
    #idx_tr = np.arange(len(arr_csi))[:600]
    #idx_te = np.arange(len(arr_csi))[600:]
        lab_ser = pd.Series(arr_lab1[:,0].astype('int'))
        lab_ser_tr = lab_ser.loc[train_index].reset_index()
        lab_ser_te = lab_ser.loc[test_index].reset_index()

        X = arr_csi1[train_index].reshape([-1,1,500,30,6])
        Xval = arr_csi1[test_index].reshape([-1,1,500,30,6])
        c = {}
        cval = {}
        Y = arr_lab1[train_index]
        Yval = arr_lab1[test_index]

        for num in np.unique(lab_ser_tr[0]):
            c[num] = list(lab_ser_tr[lab_ser_tr[0]==num].index)
        for num in np.unique(lab_ser_te[0]):
            cval[num] = list(lab_ser_te[lab_ser_te[0]==num].index)
            
        data_list.append([X,c,Xval,cval,Y,Yval])
    
    return(data_list)

class Siamese_Loader:
    """For loading batches and testing tasks to a siamese net"""
    def __init__(self, data_list, data_subsets = ["train", "val"]):
        self.data = {}
        self.categories = {}
        self.info = {}
        
        X,c,Xval,cval,Y,Yval = data_list

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
        pairs=[np.zeros((batch_size, w,h,ch)) for i in range(3)]
        #initialize vector for the targets, and make one half of it '1's, so 2nd half of batch has same class
        targets=np.zeros((batch_size,))
        targets[batch_size//2:] = 1
        for i in range(batch_size):
            category = categories[i]
            cat_key = get_key(category,c)
            cat_same = c[cat_key]
            cat_diff = list(set(range(n_classes)) - set(cat_same))
            idx_1 = rng.randint(0, n_examples)
            # anker
            category = categories[i]
            pairs[0][i,:,:,:] = X[category, idx_1].reshape(w, h, ch)
            # same class
            category_s = rng.choice(cat_same,size=1,replace=False)[0]
            pairs[1][i,:,:,:] = X[category_s, idx_1].reshape(w, h, ch)
            
            # diff class
            category_d = rng.choice(cat_diff,size=1,replace=False)[0]
            pairs[2][i,:,:,:] = X[category_d, idx_1].reshape(w, h, ch)
            
            #pick images of same class for 1st half, different for 2nd
            #if i >= batch_size // 2:
                #category_2 = category  
                #category_2 = rng.choice(cat_same,size=1,replace=False)[0]
            #else: 
                #add a random number to the category modulo n classes to ensure 2nd image has
                # ..different category
                #category_2 = (category + rng.randint(1,n_classes)) % n_classes
                #category_2 = rng.choice(cat_diff,size=1,replace=False)[0]
                
            #pairs[2][i,:,:,:] = X[category_2,idx_2]
        return pairs, targets
    
    def generate(self, batch_size, s="train"):
        """a generator for batches, so model.fit_generator can be used. """
        while True:
            pairs, targets = self.get_batch(batch_size,s)
            #pairs = self.get_batch(batch_size,s)
            yield (pairs, targets)    
            #yield (pairs)    

    def get_score(self,model,pairs):
        probs = model.predict(pairs)
        return probs
    def train(self, model, epochs, verbosity):
        model.fit_generator(self.generate(batch_size),steps_per_epoch=epochs//batch_size)


def eer_graphs(dist_truth,dist_score,pos_label):
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
    #thres_mask = thresholds <= 5*np.median(dist_score)

    th_mask = thresholds#[thres_mask]
    fpr_mask= fpr#[thres_mask]
    tpr_mask = tpr#[thres_mask]

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

## Data
#PATH = "/home/herokwon/mount_data/Data/Wi-Fi_HC/180_100/"
PATH = "D:\\Data\\WIFI\\Wi-Fi_HC\\180_100\\"

#Read Data
n_splits = 2
data_skf = ReadData(PATH,n_splits)
loss_every = 100
geteer = 1
res_eer = pd.DataFrame()

for splits in range(1):#range(len(data_skf)):
    data_list = data_skf[splits]
    #Instantiate the class
    loader = Siamese_Loader(data_list)

    #Training loop
    print("!")

    batch_size = 32
    n_iter = 5000


    weights_path = os.path.join(PATH, "weights")
    print("training")

    t1 = time.time()

    for i in range(1, n_iter+1):
            #def train(self, model, epochs, verbosity):
        #loss=loader.train(siamese_net,n_iter,True)
        (inputs,targets)=loader.get_batch(batch_size)
        loss=siamese_net.train_on_batch(inputs,targets)
        #print(loss)

        if i % loss_every == 0:
            print("iteration {}, training loss: {:.5f},".format(i,loss))

        #print("Iter:" + str(i))
    
    print("Training Time:" + str(time.time() - t1))

def make_val_triplets(x,y):
    n_samples = x.shape[0]
    xa = []
    xp = []
    xn = []
    for i in np.arange(0,n_samples-1,1):
        p_idxs = []
        for j in np.arange(i+1,n_samples,1):
            if(y[i]==y[j]):
                xa.append(x[i])
                xp.append(x[j])
                p_idxs.append(j)
        n_idxs = list(set(np.arange(0,n_samples)) - set(p_idxs))
        n_rsamp = rng.choice(n_idxs,size=len(p_idxs),replace=False)
        xn.append(x[n_rsamp])
    
    arr_xa = np.array(xa).squeeze()
    arr_xp = np.array(xp).squeeze()
    arr_xn = np.vstack(xn).squeeze()
    
    return([arr_xa,arr_xp,arr_xn])

x = Xval
y = np.array([get_key(i,cval) for i in range(len(Xval))])

pairs = make_val_triplets(x,y)
score = loader.get_score(siamese_net,pairs)