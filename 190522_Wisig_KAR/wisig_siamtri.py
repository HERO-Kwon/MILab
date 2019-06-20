import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from siam_utils import *
from siamcnn_triloss_v1 import *

%matplotlib inline


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

x = Xval
y = np.array([get_key(i,cval) for i in range(len(Xval))])

pairs = make_val_triplets(x,y)
score = loader.get_score(siamese_net,pairs)

