import tensorflow as tf
from keras.layers import Input, Conv2D, Lambda, merge, Dense, Flatten,MaxPooling2D, concatenate
from keras.models import Model, Sequential
from keras.regularizers import l2
from keras import backend as K
from keras.optimizers import SGD,Adam
from keras.losses import binary_crossentropy


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

