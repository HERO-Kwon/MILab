import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import numpy.random as rng
from keras.utils import to_categorical
from siam_utils import *
from importlib import reload
#from siamkar_v1 import *
#from karnet_v1 import KARnet
from datetime import datetime 
import socket
import tensorflow as tf
import siamkar_v2
import karnet_v1
#from siamkar_v2 import SiamKar_Loader,SiamTri_Loader,Siamese_Loader,siamtri_net,siamese_net
#from karnet_v1 import KARnet
%matplotlib inline

%matplotlib inline

host = socket.gethostname()
## Data
if(host=='pregpu-2k80'):
    PATH = "/home/herokwon/mount_data/Data/Wi-Fi_HC/180_100/"
    res_path = '/home/herokwon/Git/MILab/190522_Wisig_KAR/res/'
else:
    PATH = "D:\\Data\\WIFI\\Wi-Fi_HC\\180_100\\"
    # save path
    res_path = 'D:\\Git\\MILab\\190522_Wisig_KAR\\res\\'


col_df = (['timenow','host','model_type','rseed',
'splits','n_val','fc_size','batch_size','n_iter','alpha','lr','eer','trn_time','desc'])

try:
    res_df = pd.read_csv(res_path + 'res_df_v2.csv')
except FileNotFoundError:
    res_df = pd.DataFrame()

alpha=0.05
fc_size = 512
host = socket.gethostname()

desc = 'def_siamese_karlast1024_reloadnets'

types = [0,1,2]
seeds = [10,20,30,40,50]

for rseed  in seeds:
    for model_type in types:        
        np.random.seed(rseed)        
        n_splits = 2
        data_skf = ReadData(PATH,n_splits,rseed=rseed)
        loss_every = 100
        n_val = 10
        batch_size = 32
        n_iter = 1500
        lr = 0.0001

        for splits in range(len(data_skf)):
            now = datetime.now()
            timenow = str(now).replace(' ','_').replace(':','_')
            
            res_eer = pd.DataFrame()
            list_losstr = []
            list_losste = []
            data_list = data_skf[splits]            
            
            if model_type==0:
                #reimport
                siamkar_v2 = reload(siamkar_v2)
                karnet_v1 = reload(karnet_v1)
                from karnet_v1 import KARnet
                from siamkar_v2 import SiamKar_Loader
                from siamkar_v2 import siamtri_net
                #SiamKar_Loader = siamkar_v2.SiamKar_Loader
                #siamtri_net = siamkar_v2.siamtri_net
                #kARnet = karnet_v1.KARnet
                
                # Train KARnet
                kar_x = np.squeeze(data_list[0]).reshape([-1,500*30*6])
                kar_labels = [get_key(i,data_list[1]) for i in range(len(kar_x))]
                kar_y = to_categorical(kar_labels)

                kar_xval = np.squeeze(data_list[2]).reshape([-1,500*30*6])
                kar_labels_val = [get_key(i,data_list[3]) for i in range(len(kar_xval))]
                kar_yval = to_categorical(kar_labels_val)

                #kar = KARnet([kar_x,kar_y,kar_xval,kar_yval],h=[2048,1024,fc_size],rseed=rseed)
                kar = KARnet([kar_x,kar_y,kar_xval,kar_yval],h=[1024,fc_size],rseed=rseed)
                #kar = KARnet([kar_x,kar_y,kar_xval,kar_yval],h=[8192,fc_size],rseed=rseed)
                t1 = time.time()
                kar.train()
                print("KAR Training time : "+ str(time.time()-t1))

                #Instantiate the class
                loader = SiamKar_Loader(data_list,kar,lr,host)
            elif model_type==1:
                #reimport
                siamkar_v2 = reload(siamkar_v2)
                from siamkar_v2 import SiamTri_Loader
                from siamkar_v2 import siamtri_net
                #SiamTri_Loader = siamkar_v2.SiamTri_Loader
                #siamtri_net = siamkar_v2.siamtri_net
                #Instantiate the class
                loader = SiamTri_Loader(data_list,lr,host)#,kar,dist_rate)
            elif model_type==2:
                #reimport
                siamkar_v2 = reload(siamkar_v2)
                from siamkar_v2 import Siamese_Loader
                from siamkar_v2 import siamese_net
                #Siamese_Loader = siamkar_v2.Siamese_Loader
                #siamese_net = siamkar_v2.siamese_net
                loader = Siamese_Loader(data_list,lr,host)

            #Training loop
            print("!")
            print("training")

            t1 = time.time()

            for i in range(0, n_iter):
                #loss=loader.train(siamtri_net,n_epoch,batch_size,True)
                (inputs,targets)=loader.get_batch(batch_size,s='train')
                loss=loader.net.train_on_batch(inputs,targets)
                #print(loss)
                list_losstr.append([splits,loss])

                if i % loss_every == 0:
                    (x_val,y_val)=loader.get_batch(n_val,s='val')
                    loss_te = loader.net.test_on_batch(x_val, y_val, sample_weight=None)
                    list_losste.append([splits,loss_te])
                    print("iteration {}, training loss: {:.5f}, validation loss: {:.5f}".format(i,loss,loss_te))

            trn_time = time.time()-t1
            print("Training Time:" + str(trn_time))

            # figures
            save_path = res_path + timenow + '_' + str(splits)
            #print("Iter:" + str(i))
            fig = plt.figure()
            ltr = [list_losstr[j][1] for j in range(len(list_losstr)) if list_losstr[j][0] == splits]
            lte = [list_losste[j][1] for j in range(len(list_losste)) if list_losste[j][0] == splits]
            plt.plot(list(np.arange(0,len(ltr),1)),ltr)
            plt.plot(list(np.arange(0,100*len(lte),100)),lte)
            plt.title('loss plot')
            #plt.ylim([0,0.40])
            plt.show()
            fig.savefig(save_path + '_fig_loss.png')
            plt.close(fig)

            ser_loss = pd.Series({'ltr':ltr,'lte':lte})
            ser_loss.to_csv(save_path + '_loss.csv')


            # Make Pairs   
            Xval = data_list[2]
            cval = data_list[3]
            xte = Xval
            yte = np.array([get_key(i,cval) for i in range(len(Xval))])
            cte = cval

            gen_pairs = gen_val_triplets(xte,yte,cte)

            if ((model_type==0)|(model_type==1)):
                # Triplet
                list_score = []
                for i in range(xte.shape[0]):
                    try:
                        pairs = next(gen_pairs)
                        score = loader.get_score(loader.net,pairs)
                        list_score.append(score)
                    except:
                        break
                arr_score = np.vstack(list_score)

                sc_ank = arr_score[:,:fc_size]
                sc_pos = arr_score[:,fc_size:2*fc_size]
                sc_neg = arr_score[:,2*fc_size:]
                l2_pos = np.linalg.norm(sc_ank-sc_pos,ord=2,axis=1)
                l2_neg = np.linalg.norm(sc_ank-sc_neg,ord=2,axis=1)

                l2_dist = np.concatenate([l2_pos,l2_neg])
                l2_label = np.concatenate([np.ones(shape=l2_pos.shape),np.zeros(shape=l2_neg.shape)])
                eer = eer_graphs(l2_label,l2_dist,0,save_path=save_path)


            elif model_type==2:
                # Siamese
                list_score = []
                list_target = []
                for i in range(xte.shape[0]):
                    try:
                        [arr_xa,arr_xp,arr_xn] = next(gen_pairs)
                        pairs_ank = np.tile(arr_xa,(2,1,1,1))
                        pairs_pn = np.vstack([arr_xp,arr_xn])
                        pairs = [pairs_ank,pairs_pn]
                        targets=np.ones(len(pairs_pn)).reshape([-1,1])
                        targets[len(pairs_pn)//2:] = 0 
                        score = loader.get_score(loader.net,pairs)
                        list_score.append(score)
                        list_target.append(targets)
                    except:
                        break
                arr_score = np.vstack(list_score).squeeze()
                arr_target = np.vstack(list_target).squeeze()

                eer = eer_graphs(arr_target,arr_score,1,save_path=save_path)
                
                #tf.keras.backend.clear_session()
            
            del(loader.net)
            del(loader)
            
            #save results
            res_ser =  pd.DataFrame([[timenow,host,model_type,rseed,splits,
            n_val,fc_size,batch_size,n_iter,alpha,lr,eer,trn_time,desc]],columns=col_df)
            res_df = res_df.append(res_ser,ignore_index=True)  
            res_df.to_csv(res_path + 'res_df_v2.csv',index=False)