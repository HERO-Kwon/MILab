import tensorflow as tf
from keras.layers import Input, Conv2D, Lambda, merge, Dense, Flatten,MaxPooling2D, concatenate
from keras.models import Model, Sequential
from keras.regularizers import l2
from keras import backend as K
from keras.optimizers import SGD,Adam
from keras.losses import binary_crossentropy
from siam_utils import get_key
import numpy.random as rng
import numpy as np
from keras.utils.multi_gpu_utils import multi_gpu_model

fc_size = 512

def triplet_loss(y_true, y_pred):
    #fc_size = 128
    alpha = 0.05
    a_pred = y_pred[:, 0:fc_size]
    p_pred = y_pred[:, fc_size:2*fc_size]
    n_pred = y_pred[:, 2*fc_size:3*fc_size]
    pos_dist = tf.reduce_sum(tf.square(tf.subtract(a_pred, p_pred)), axis=-1)
    neg_dist = tf.reduce_sum(tf.square(tf.subtract(a_pred, n_pred)), axis=-1)
    basic_loss = tf.add(tf.subtract(pos_dist, neg_dist), alpha)
    loss = tf.reduce_mean(tf.maximum(basic_loss, 0.0),0)
    #basic_loss = tf.add(tf.subtract(pos_dist, neg_dist), fc_size)
    #loss = tf.reduce_sum(basic_loss)
    
    return loss
def W_init(shape,name=None):
    """Initialize weights as in paper"""
    values = rng.normal(loc=0,scale=1e-2,size=shape)
    return K.variable(values,name=name)
#//TODO: figure out how to initialize layer biases in keras.
def b_init(shape,name=None):
    """Initialize bias as in paper"""
    values=rng.normal(loc=0.5,scale=1e-2,size=shape)
    return K.variable(values,name=name)
# Network: Siamese with Triplet loss
input_shape = (500, 30, 6)
ank_input = Input(input_shape)
pos_input = Input(input_shape)
neg_input = Input(input_shape)
#build convnet to use in each siamese 'leg'

convnet = Sequential()
convnet.add(Conv2D(64,(3,3),activation='relu',input_shape=input_shape,padding='same',
                   kernel_initializer=W_init,kernel_regularizer=l2(2e-4)))
convnet.add(MaxPooling2D())
convnet.add(Conv2D(128,(3,3),activation='relu',padding='same',
                   kernel_regularizer=l2(2e-4),kernel_initializer=W_init,bias_initializer=b_init))
convnet.add(MaxPooling2D())
convnet.add(Conv2D(256,(3,3),activation='relu',padding='same',
                   kernel_regularizer=l2(2e-4),kernel_initializer=W_init,bias_initializer=b_init))
convnet.add(MaxPooling2D())
'''
convnet.add(Conv2D(512,(3,3),activation='relu',padding='same',
                   kernel_regularizer=l2(2e-4),kernel_initializer=W_init,bias_initializer=b_init))
convnet.add(MaxPooling2D())
'''
#convnet.add(MaxPooling2D(2,2))
convnet.add(Flatten())
convnet.add(Dense(fc_size,activation="sigmoid",
                  kernel_regularizer=l2(1e-3),kernel_initializer=W_init,bias_initializer=b_init))

'''
convnet = Sequential()
convnet.add(Conv2D(64,(3,3),activation='relu',input_shape=input_shape, padding='same'))
                    #kernel_regularizer=l2(1e-4)))
                    #kernel_initializer=W_init,
convnet.add(MaxPooling2D())
convnet.add(Conv2D(128,(3,3),activation='relu', padding='same'))
                    #kernel_regularizer=l2(1e-4)))#,kernel_initializer=W_init,bias_initializer=b_init))
convnet.add(MaxPooling2D())
convnet.add(Conv2D(256,(3,3),activation='relu', padding='same'))
                    #kernel_regularizer=l2(1e-4)))#,kernel_initializer=W_init,bias_initializer=b_init))
convnet.add(MaxPooling2D())
convnet.add(Conv2D(512,(3,3),activation='relu', padding='same'))
                    #kernel_regularizer=l2(1e-4)))#,kernel_initializer=W_init,bias_initializer=b_init))
convnet.add(MaxPooling2D())
#convnet.add(MaxPooling2D(2,2))
convnet.add(Flatten())
#convnet.add(Dense(1024,activation="sigmoid"))#,kernel_initializer=W_init,bias_initializer=b_init))
convnet.add(Dense(1024,activation="sigmoid"))#,kernel_initializer=W_init,bias_initializer=b_init)
'''

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
siamtri_net = Model(inputs=[ank_input, pos_input, neg_input],outputs=merged_vector)


#optimizer = Adam(0.0006)#Adam(0.1)
#sgd = SGD(lr=1e-5, momentum=0.9, nesterov=True, decay=1e-6)
#//TODO: get layerwise learning rates and momentum annealing scheme described in paperworking
#siamtri_net.compile(loss=triplet_loss,optimizer=optimizer)
#siamtri_net.summary()

input_shape = (500, 30, 6)
left_input = Input(input_shape)
right_input = Input(input_shape)
#build convnet to use in each siamese 'leg'

convnet = Sequential()
convnet.add(Conv2D(64,(3,3),activation='relu',input_shape=input_shape,padding='same',
                   kernel_initializer=W_init,kernel_regularizer=l2(2e-4)))
convnet.add(MaxPooling2D())
convnet.add(Conv2D(128,(3,3),activation='relu',padding='same',
                   kernel_regularizer=l2(2e-4),kernel_initializer=W_init,bias_initializer=b_init))
convnet.add(MaxPooling2D())
convnet.add(Conv2D(256,(3,3),activation='relu',padding='same',
                   kernel_regularizer=l2(2e-4),kernel_initializer=W_init,bias_initializer=b_init))
convnet.add(MaxPooling2D())
'''
convnet.add(Conv2D(512,(3,3),activation='relu',padding='same',
                   kernel_regularizer=l2(2e-4),kernel_initializer=W_init,bias_initializer=b_init))
convnet.add(MaxPooling2D())
'''
#convnet.add(MaxPooling2D(2,2))
convnet.add(Flatten())
convnet.add(Dense(fc_size,activation="sigmoid",kernel_regularizer=l2(1e-3),kernel_initializer=W_init,bias_initializer=b_init))
'''
convnet = Sequential()
convnet.add(Conv2D(64,(3,3),activation='relu',input_shape=input_shape, padding='same'))
                    #kernel_regularizer=l2(1e-4)))
                    #kernel_initializer=W_init,
convnet.add(MaxPooling2D())
convnet.add(Conv2D(128,(3,3),activation='relu', padding='same'))
                    #kernel_regularizer=l2(1e-4)))#,kernel_initializer=W_init,bias_initializer=b_init))
convnet.add(MaxPooling2D())
convnet.add(Conv2D(256,(3,3),activation='relu', padding='same'))
                    #kernel_regularizer=l2(1e-4)))#,kernel_initializer=W_init,bias_initializer=b_init))
convnet.add(MaxPooling2D())
convnet.add(Conv2D(512,(3,3),activation='relu', padding='same'))
                    #kernel_regularizer=l2(1e-4)))#,kernel_initializer=W_init,bias_initializer=b_init))
convnet.add(MaxPooling2D())
#convnet.add(MaxPooling2D(2,2))
convnet.add(Flatten())
#convnet.add(Dense(1024,activation="sigmoid"))#,kernel_initializer=W_init,bias_initializer=b_init))
convnet.add(Dense(1024,activation="sigmoid"))#,kernel_initializer=W_init,bias_initializer=b_init)
'''

#call the convnet Sequential model on each of the input tensors so params will be shared
encoded_l = convnet(left_input)
encoded_r = convnet(right_input)
#layer to merge two encoded inputs with the l1 distance between them
L1_layer = Lambda(lambda tensors:K.abs(tensors[0] - tensors[1]))
#call this layer on list of two input tensors.
L1_distance = L1_layer([encoded_l, encoded_r])
prediction = Dense(1,activation='sigmoid',bias_initializer=b_init)(L1_distance)
siamese_net = Model(inputs=[left_input,right_input],outputs=prediction)

#optimizer = Adam(0.00006)
#//TODO: get layerwise learning rates and momentum annealing scheme described in paperworking
#siamese_net.compile(loss="binary_crossentropy",optimizer=optimizer)
#siamese_net.count_params()



class SiamTri_Loader:
    """For loading batches and testing tasks to a siamese net"""
    def __init__(self, data_list,lr,host,data_subsets = ["train", "val"]):
        self.data = {}
        self.categories = {}
        self.info = {}
        
        X_d,c_d,Xval_d,cval_d,Y_d,Yval_d = data_list

        self.data["train"] = X_d
        self.data["val"] = Xval_d
        self.categories["train"] = c_d
        self.categories["val"] = cval_d
        self.host = host
        self.lr = lr
        
        if(host=='pregpu-2k80'):
            self.net = multi_gpu_model(siamtri_net, gpus=2)
        else:
            self.net = siamtri_net
        self.net.compile(loss=triplet_loss,optimizer=Adam(self.lr))
        self.net.summary()

    def get_batch(self,batch_size,s):
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

    def generate(self, batch_size,s):
        """a generator for batches, so model.fit_generator can be used. """
        while True:
            pairs, targets = self.get_batch(batch_size,s)
            #pairs = self.get_batch(batch_size,s)
            yield (pairs, targets)    
            #yield (pairs)    

    def get_score(self,model,pairs):
        probs = model.predict(pairs)
        return probs
    def train(self, model, epochs,batch_size, verbosity):
        model.fit_generator(self.generate(batch_size),steps_per_epoch=epochs//batch_size)

class SiamKar_Loader(SiamTri_Loader):
    def __init__(self,data_list,kar,lr,host,data_subsets = ["train", "val"]):
        super(SiamKar_Loader, self).__init__(data_list,lr,host)
        #super(SiamKar_Loader, self).__init__(net)
        #super(SiamKar_Loader, self).__init__(lr)
        #super(SiamKar_Loader, self).__init__(alpha)
        self.kar = kar
        self.kar_fc = self.dist_kar()
        #self.dist_rate = dist_rate
        #self.lr = lr
        #self.alpha = alpha
        X_d,c_d,Xval_d,cval_d,Y_d,Yval_d = data_list

        self.data["train"] = X_d
        self.data["val"] = Xval_d
        self.categories["train"] = c_d
        self.categories["val"] = cval_d
        
        #self.net = Super().net#siamtri_net
        #self.net.compile(loss=triplet_loss,optimizer=Adam(self.lr))
        #self.net.summary()


    def dist_kar(self):
        kar_x = np.squeeze(self.data["train"]).reshape([-1,500*30*6])
        bias = np.ones([kar_x.shape[0],1])
        x_calc = kar_x
        #for i in range(len(self.kar.W)-1):
        for i in range(len(self.kar.W)):
            x_calc = 1*self.kar.f(np.hstack([bias,x_calc]).dot(self.kar.W[i]))
        kar_fc = x_calc
        return(kar_fc)

    def get_batch(self,batch_size,s):
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

        for i in range(batch_size):
            category = categories[i]
            cat_key = get_key(category,c)
            cat_same = list(set(c[cat_key]) - set([category]))
            cat_diff = list(set(range(n_classes)) - set(cat_same) - set([category]))
            #cat_diff_s = np.random.choice(cat_diff,int(self.dist_rate*len(cat_diff)))
            
            idx_1 = rng.randint(0, n_examples)
            # anker
            category = categories[i]
            pairs[0][i,:,:,:] = X[category, idx_1].reshape(w, h, ch)
            # same class
            category_s = rng.choice(cat_same,size=1,replace=False)[0]
            pairs[1][i,:,:,:] = X[category_s, idx_1].reshape(w, h, ch)
            
            # diff class
            category_d  = rng.choice(cat_diff,size=1,replace=False)[0]
            diff_key = get_key(category_d,c)
            cat_diff_s = c[diff_key]
            
            # Kar distance
            if(s=='train'):
                kar_ank = self.kar_fc[category]
                kar_diff = self.kar_fc[cat_diff_s]
                dist_diff = np.linalg.norm([kar_diff - kar_ank],ord=2,axis=-1).squeeze()
                # select hard samples
                dist_hard = np.percentile(dist_diff, 25, interpolation='nearest')
                hard_samples = dist_diff <= dist_hard
                
                cat_diff_hard = list(np.array(cat_diff_s)[hard_samples])

                category_d = rng.choice(cat_diff_hard,size=1,replace=False)[0]
                pairs[2][i,:,:,:] = X[category_d, idx_1].reshape(w, h, ch)
            else:
                #category_d = rng.choice(cat_diff,size=1,replace=False)[0]
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

class Siamese_Loader:
    """For loading batches and testing tasks to a siamese net"""
    def __init__(self, data_list, lr,host,data_subsets = ["train", "val"]):
        self.data = {}
        self.categories = {}
        self.info = {}
        
        X_d,c_d,Xval_d,cval_d,Y_d,Yval_d = data_list

        self.data["train"] = X_d
        self.data["val"] = Xval_d
        self.categories["train"] = c_d
        self.categories["val"] = cval_d
        
        self.lr = lr

        if(host=='pregpu-2k80'):
            self.net = multi_gpu_model(siamese_net, gpus=2)
        else:
            self.net = siamese_net
        self.net.compile(loss="binary_crossentropy",optimizer=Adam(self.lr))
        self.net.count_params()
    def get_batch(self,batch_size,s):
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
    
    def generate(self, batch_size,s):
        """a generator for batches, so model.fit_generator can be used. """
        while True:
            pairs, targets = self.get_batch(batch_size,s)
            yield (pairs, targets)    
    def get_score(self,model,pairs):
        probs = model.predict(pairs)
        return probs
    '''
    def make_oneshot_task(self,N,s="val",language=None):
        """Create pairs of test image, support set for testing N way one-shot learning. """
        X=self.data[s]
        c=self.categories[s]
        n_classes, n_examples, w, h,ch = X.shape
        indices = rng.randint(0,n_examples,size=(N,))

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
    def train(self, model, epochs, verbosity):
        model.fit_generator(self.generate(batch_size),)
    '''