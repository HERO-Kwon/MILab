from keras.layers import Input, Conv2D, Lambda, merge, Dense, Flatten,MaxPooling2D
from keras.models import Model, Sequential
from keras.regularizers import l2
from keras import backend as K
from keras.optimizers import SGD,Adam
from keras.losses import binary_crossentropy
import numpy.random as rng
import numpy as np
import os
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.utils import shuffle
import pandas as pd
from sklearn.model_selection import train_test_split

%matplotlib inline
def W_init(shape,name=None):
    """Initialize weights as in paper"""
    values = rng.normal(loc=0,scale=1e-2,size=shape)
    return K.variable(values,name=name)
#//TODO: figure out how to initialize layer biases in keras.
def b_init(shape,name=None):
    """Initialize bias as in paper"""
    values=rng.normal(loc=0.5,scale=1e-2,size=shape)
    return K.variable(values,name=name)


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
convnet.add(MaxPooling2D())
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

optimizer = Adam(0.00006)
#//TODO: get layerwise learning rates and momentum annealing scheme described in paperworking
siamese_net.compile(loss="binary_crossentropy",optimizer=optimizer)

siamese_net.count_params()

# function to return key for any value 
def get_key(val,my_dict): 
    for key, list_val in my_dict.items(): 
         if val in list_val: 
             return key 
  
    return "key doesn't exist"

## Data
PATH = "/home/herokwon/mount_data/Data/Wi-Fi_HC/180_100/"

def ReadData(PATH):
    list_csi = []
    list_lab = []

    file_list = os.listdir(PATH)
    file_list = [f for f in file_list if '.npy' in f]
    
    for file in file_list:
        data_read = np.load(PATH + file)
        csi_read = data_read[:,4:].reshape([-1,10,6,30,500]).swapaxes(2,4)
        lab_read = data_read[:,0].reshape([-1,10])
        
        list_csi.append(csi_read)
        list_lab.append(lab_read)
        
    arr_csi = np.array(list_csi).swapaxes(0,1).reshape([100*8,10,500,30,6])
    arr_lab = np.array(list_lab).swapaxes(0,1).reshape([100*8,10])

    idx_tr,idx_te = train_test_split(np.arange(len(arr_csi)),test_size=0.2,random_state=10)
    #idx_tr = np.arange(len(arr_csi))[:600]
    #idx_te = np.arange(len(arr_csi))[600:]
    lab_ser = pd.Series(arr_lab[:,0].astype('int'))
    lab_ser_tr = lab_ser.loc[idx_tr].reset_index()
    lab_ser_te = lab_ser.loc[idx_te].reset_index()

    #filter outliers
    xs_median = np.median(arr_csi)
    
    
    X = arr_csi[idx_tr] / xs_median
    Xval = arr_csi[idx_te] / xs_median
    c = {}
    cval = {}
    
    for num in np.unique(lab_ser_tr[0]):
        c[num] = list(lab_ser_tr[lab_ser_tr[0]==num].index)
    for num in np.unique(lab_ser_te[0]):
        cval[num] = list(lab_ser_te[lab_ser_te[0]==num].index)

    print("training id")
    print(c.keys())
    print("validation id:")
    print(cval.keys())
    
    return(X,c,Xval,cval)

X,c,Xval,cval = ReadData(PATH)


class Siamese_Loader:
    """For loading batches and testing tasks to a siamese net"""
    def __init__(self, path, data_subsets = ["train", "val"]):
        self.data = {}
        self.categories = {}
        self.info = {}
        
        print("loading data from {}".format(path))
        X,c,Xval,cval = ReadData(path)

        self.data["train"] = X
        self.data["val"] = Xval
        self.categories["train"] = c
        self.categories["val"] = cval
    def get_batch(self,batch_size,s="train"):
        """Create batch of n pairs, half same class, half different class"""
        X=self.data[s]
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

    def make_oneshot_task(self,N,s="val",language=None):
        """Create pairs of test image, support set for testing N way one-shot learning. """
        X=self.data[s]
        n_classes, n_examples, w, h,ch = X.shape
        indices = rng.randint(0,n_examples,size=(N,))
        if language is not None:
            low, high = self.categories[s][language]
            if N > high - low:
                raise ValueError("This language ({}) has less than {} letters".format(language, N))
            categories = rng.choice(range(low,high),size=(N,),replace=False)
            
        else:#if no language specified just pick a bunch of random letters
            categories = rng.choice(range(n_classes),size=(N,),replace=False)            
        true_category = categories[0]
        ex1, ex2 = rng.choice(n_examples,replace=False,size=(2,))
        test_image = np.asarray([X[true_category,ex1,:,:]]*N).reshape(N, w, h,ch)
        support_set = X[categories,indices,:,:]
        support_set[0,:,:] = X[true_category,ex2]
        support_set = support_set.reshape(N, w, h,ch)
        targets = np.zeros((N,))
        targets[0] = 1
        targets, test_image, support_set = shuffle(targets, test_image, support_set)
        pairs = [test_image,support_set]

        return pairs, targets
    
    def test_oneshot(self,model,N,k,s="val",verbose=0):
        """Test average N way oneshot learning accuracy of a siamese neural net over k one-shot tasks"""
        n_correct = 0
        if verbose:
            print("Evaluating model on {} random {} way one-shot learning tasks ...".format(k,N))
        for i in range(k):
            inputs, targets = self.make_oneshot_task(N,s)
            probs = model.predict(inputs)
            if np.argmax(probs) == np.argmax(targets):
                n_correct+=1
        percent_correct = (100.0*n_correct / k)
        if verbose:
            print("Got an average of {}% {} way one-shot learning accuracy".format(percent_correct,N))
        return percent_correct
    def get_score(self,model,pairs):
        probs = model.predict(pairs)
        return probs
    def train(self, model, epochs, verbosity):
        model.fit_generator(self.generate(batch_size),
                            
                             )
#Instantiate the class
loader = Siamese_Loader(PATH)



#Training loop
print("!")
evaluate_every = 10 # interval for evaluating on one-shot tasks
loss_every=50 # interval for printing loss (iterations)
batch_size = 32
n_iter = 100
N_way = 50 #20 # how many classes for testing one-shot tasks>
n_val = 250 #how mahy one-shot tasks to validate on?
best = -1
weights_path = os.path.join(PATH, "weights")
print("training")
import time
for i in range(1, n_iter+1):
    t1 = time.time()
    (inputs,targets)=loader.get_batch(batch_size)
    loss=siamese_net.train_on_batch(inputs,targets)
    print(loss)
    if i % evaluate_every == 0:
        print("evaluating")
        val_acc = loader.test_oneshot(siamese_net,N_way,n_val,verbose=True)
        if True: #val_acc >= best:
            print(i)
            print("saving")
            siamese_net.save(weights_path)
            best=val_acc

    if i % loss_every == 0:
        print("iteration {}, training loss: {:.2f},".format(i,loss))
    
    print(i)
    print(time.time() - t1)

def nearest_neighbour_correct(pairs,targets):
    """returns 1 if nearest neighbour gets the correct answer for a one-shot task
        given by (pairs, targets)"""
    L2_distances = np.zeros_like(targets)
    for i in range(len(targets)):
        L2_distances[i] = np.sum(np.sqrt(pairs[0][i]**2 - pairs[1][i]**2))
    if np.argmin(L2_distances) == np.argmax(targets):
        return 1
    return 0


def test_nn_accuracy(N_ways,n_trials,loader):
    """Returns accuracy of one shot """
    print("Evaluating nearest neighbour on {} unique {} way one-shot learning tasks ...".format(n_trials,N_ways))

    n_right = 0
    
    for i in range(n_trials):
        pairs,targets = loader.make_oneshot_task(N_ways,"val")
        correct = nearest_neighbour_correct(pairs,targets)
        n_right += correct
    return 100.0 * n_right / n_trials


ways = np.arange(1, 20, 2)#np.arange(1, 60, 2)
resume =  False
val_accs, train_accs,nn_accs = [], [], []
trials = 450
for N in ways:
    val_accs.append(loader.test_oneshot(siamese_net, N,trials, "val", verbose=True))
    train_accs.append(loader.test_oneshot(siamese_net, N,trials, "train", verbose=True))
    nn_accs.append(test_nn_accuracy(N,trials, loader))
    
#plot the accuracy vs num categories for each
plt.plot(ways, val_accs, "m")
plt.plot(ways, train_accs, "y")
plt.plot(ways, nn_accs, "c")

plt.plot(ways,100.0/ways,"r")
plt.show()


fig,ax = plt.subplots(1)
ax.plot(ways,val_accs,"m",label="Siamese(val set)")
ax.plot(ways,train_accs,"y",label="Siamese(train set)")
plt.plot(ways,nn_accs,label="Nearest neighbour")

ax.plot(ways,100.0/ways,"g",label="Random guessing")
plt.xlabel("Number of possible classes in one-shot tasks")
plt.ylabel("% Accuracy")
plt.title("Omiglot One-Shot Learning Performance of a Siamese Network")
box = ax.get_position()
ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
inputs,targets = loader.make_oneshot_task(20,"val")
plt.show()

print(inputs[0].shape)
#plot_oneshot_task(inputs)
p=siamese_net.predict(inputs)
print(p)


## making EER

#make eer
Xval.shape
n_classes, n_examples, h, w, ch = Xval.shape

import time
res_list = []
for i in range(n_classes):
    #for j in range(n_examples):
    t1 =time.time()
    test_image = Xval[i,0].reshape([-1,500,30,6])
        
    for ti in range(n_classes):
        #for tj in range(n_examples):
        support_image = Xval[ti,0].reshape([-1,500,30,6])
        pairs = [test_image, support_image]
        score = loader.get_score(siamese_net,pairs)
        
        score_list = [i,ti,int(get_key(i,cval) == get_key(ti,cval)),score[0][0]]
        res_list.append(score_list)
    print(i)
    print(time.time() - t1)

res_df = pd.DataFrame(res_list,columns=['img1','img2','TF','score'])
res_df.head()
res_df1 = res_df[res_df.img1 != res_df.img2]

## Function : EER Curve
import random
from sklearn.metrics import confusion_matrix
def EER_Curve(tf_list,err_values,sampling_thres):

    print("EER Curve")

    eer_df = pd.DataFrame(columns = ['thres','fn','fp'])
    
    n_thres = int(sampling_thres*len(err_values))
    sampled_err_values = random.sample(err_values,n_thres)

    for i, thres in enumerate(set(sampled_err_values)):
        predicted_tf = [e <= thres for e in err_values]
        
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

res_eer = EER_Curve(list(res_df1.TF),list(res_df1.score),0.01)