import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import numpy.random as rng
from keras.utils import to_categorical

from siam_utils import *
from siamkar_v1 import Siamese_Loader,siamese_net
#from karnet_v1 import KARnet

np.random.seed(8)
%matplotlib inline

## Data
PATH = "/home/herokwon/mount_data/Data/Wi-Fi_HC/180_100/"
#PATH = "D:\\Data\\WIFI\\Wi-Fi_HC\\180_100\\"

#Read Data
n_splits = 2
data_skf = ReadData(PATH,n_splits)
loss_every = 100
n_val = 10
dist_rate= 0.1
batch_size = 32
n_iter = 4096

res_eer = pd.DataFrame()

list_losstr = []
list_losste = []
for splits in range(len(data_skf)):
    data_list = data_skf[splits]
    '''
    # Train KARnet
    kar_x = np.squeeze(data_list[0]).reshape([-1,500*30*6])
    kar_labels = [get_key(i,data_list[1]) for i in range(len(kar_x))]
    kar_y = to_categorical(kar_labels)

    kar_xval = np.squeeze(data_list[2]).reshape([-1,500*30*6])
    kar_labels_val = [get_key(i,data_list[3]) for i in range(len(kar_xval))]
    kar_yval = to_categorical(kar_labels_val)


    kar = KARnet([kar_x,kar_y,kar_xval,kar_yval],h=[2048,1024],rseed=8)
    t1 = time.time()
    kar.train()
    print("KAR Training time : "+ str(time.time()-t1))
    '''

    #Instantiate the class
    #loader = SiamKar_Loader(data_list,kar,dist_rate)
    loader = Siamese_Loader(data_list)

    #Training loop
    print("!")
    print("training")

    t1 = time.time()
    
    for i in range(0, n_iter):
        #loss=loader.train(siamese_net,n_epoch,batch_size,True)
        (inputs,targets)=loader.get_batch(batch_size,s='train')
        loss=siamese_net.train_on_batch(inputs,targets)
        #print(loss)
        list_losstr.append([splits,loss])

        if i % loss_every == 0:
            (x_te,y_te)=loader.get_batch(n_val,s='val')
            loss_te = siamese_net.test_on_batch(x_te, y_te, sample_weight=None)
            list_losste.append([splits,loss_te])
            print("iteration {}, training loss: {:.5f}, validation loss: {:.5f}".format(i,loss,loss_te))
            
    #print("Iter:" + str(i))
    fig = plt.figure()
    ltr = [list_losstr[i][1] for i in range(len(list_losstr)) if list_losstr[i][0] == splits]
    lte = [list_losste[i][1] for i in range(len(list_losste)) if list_losste[i][0] == splits]
    plt.plot(list(np.arange(0,len(ltr),1)),ltr)
    plt.plot(list(np.arange(0,100*len(lte),100)),lte)
    plt.title('loss plot')
    plt.show()
    plt.close(fig)
    print("Training Time:" + str(time.time() - t1))
    
    Xval = data_list[2]
    cval = data_list[3]
    x = Xval
    y = np.array([get_key(i,cval) for i in range(len(Xval))])

    pairs = make_val_pairs(x,y)
    score = loader.get_score(siamese_net,pairs)
    del(pairs)
    targets=np.ones(len(score))
    targets[len(score)//2:] = 0
    '''
    sc_ank = score[:,:1024]
    sc_pos = score[:,1024:2048]
    sc_neg = score[:,2048:]
    l2_pos = np.linalg.norm(sc_ank-sc_pos,ord=2,axis=1)
    l2_neg = np.linalg.norm(sc_ank-sc_neg,ord=2,axis=1)

    l2_dist = np.concatenate([l2_pos,l2_neg])
    l2_label = np.concatenate([np.ones(shape=l2_pos.shape),np.zeros(shape=l2_neg.shape)])
    eer_graphs(l2_label,l2_dist,0)
    '''
    
    eer_graph = eer_graphs(targets,1-score,0)