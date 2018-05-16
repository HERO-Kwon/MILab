import os
import sys
import argparse
import numpy as np

#from ..utils import subsample, load_image_data, multi_gpu_model, get_image_file_paths, create_output_dir
#from ..utils.constants import SLICE_WIDTH, SLICE_HEIGHT

from datetime import datetime

from keras.models import Model
from keras.layers import Input, Dense, Activation, concatenate, UpSampling2D
from keras.layers.convolutional import Conv2D, Conv2DTranspose
from keras.layers.pooling import MaxPooling2D, AveragePooling2D
from keras.losses import mean_squared_error, mean_absolute_error
from keras.optimizers import RMSprop
from keras.initializers import RandomNormal
from keras.callbacks import ModelCheckpoint


# Training set construction
NUM_SAMPLE_SLICES = 35

# Neural Network Parameters
RMS_WEIGHT_DECAY = .9
LEARNING_RATE = .001
FNET_ERROR_MSE = "mse"
FNET_ERROR_MAE = "mae"

# Checkpointing
CHECKPOINT_FILE_PATH_FORMAT = "fnet-{epoch:02d}.hdf5"
SFX_NETWORK_CHECKPOINTS = "checkpoints"

class FNet:
    def __init__(self, num_gpus, error):
        self.architecture_exists = False
        self.num_gpus = num_gpus
        self.error = error

    def train(self, y_folded, y_original, batch_size, num_epochs):
        """
        Trains the specialized U-net for the MRI reconstruction task

        Parameters
        ------------
        y_folded : [np.ndarray]
            A set of folded images obtained by subsampling k-space data
        y_original : [np.ndarray]
            The ground truth set of images, preprocessed by applying the inverse
            f_{cor} function and removing undersampled k-space data
        batch_size : int
            The training batch size
        num_epochs : int
            The number of training epochs
        checkpoints_dir : str
            The base directory under which to store network checkpoints 
            after each iteration
        """

        if not self.architecture_exists:
            self._create_architecture()

        checkpoints_dir =  os.getcwd() + '\\chkpnt'
        checkpoints_dir_path = create_output_dir(base_path=checkpoints_dir,
                                                 suffix=SFX_NETWORK_CHECKPOINTS,
                                                 exp_name=None)
        checkpoint_fpath_format = os.path.join(checkpoints_dir_path, CHECKPOINT_FILE_PATH_FORMAT)
        checkpoint_callback = ModelCheckpoint(
            checkpoint_fpath_format, monitor='val_loss', period=1)

        print("Network checkpoints will be saved to: '{}'".format(checkpoints_dir_path))

        self.model.fit(
            y_folded,
            y_original,
            batch_size=batch_size,
            epochs=num_epochs,
            shuffle=True,
            validation_split=.2,
            verbose=1,
            callbacks=[checkpoint_callback])

    def _parse_error(self):
        if self.error == FNET_ERROR_MSE:
            return mean_squared_error
        elif self.error == FNET_ERROR_MAE:
            return mean_absolute_error
        else:
            raise Exception(
                "Attempted to train network with an invalid loss function!")

    def _get_initializer_seed(self):
        epoch = datetime.utcfromtimestamp(0)
        curr_time = datetime.now()
        millis_since_epoch = (curr_time - epoch).total_seconds() * 1000
        return int(millis_since_epoch)

    def _create_architecture(self):
        inputs = Input(shape=(256, 256, 1))

        weights_initializer = RandomNormal(
            mean=0.0, stddev=.01, seed=self._get_initializer_seed())

        # Using the padding=`same` option is equivalent to zero padding
        conv2d_1 = Conv2D(
            filters=64,
            kernel_size=(3, 3),
            strides=(1, 1),
            padding='same',
            activation='relu',
            kernel_initializer=weights_initializer)(inputs)

        conv2d_2 = Conv2D(
            filters=64,
            kernel_size=(3, 3),
            strides=(1, 1),
            padding='same',
            activation='relu',
            kernel_initializer=weights_initializer)(conv2d_1)

        maxpool_1 = MaxPooling2D(
            pool_size=(2, 2), strides=(2, 2), padding='same')(conv2d_2)

        conv2d_3 = Conv2D(
            filters=128,
            kernel_size=(3, 3),
            strides=(1, 1),
            padding='same',
            activation='relu',
            kernel_initializer=weights_initializer)(maxpool_1)

        conv2d_4 = Conv2D(
            filters=128,
            kernel_size=(3, 3),
            strides=(1, 1),
            padding='same',
            activation='relu',
            kernel_initializer=weights_initializer)(conv2d_3)

        maxpool_2 = MaxPooling2D(
            pool_size=(2, 2), strides=(2, 2), padding='same')(conv2d_4)

        conv2d_5 = Conv2D(
            filters=256,
            kernel_size=(3, 3),
            strides=(1, 1),
            padding='same',
            activation='relu',
            kernel_initializer=weights_initializer)(maxpool_2)

        conv2d_6 = Conv2D(
            filters=128,
            kernel_size=(3, 3),
            strides=(1, 1),
            padding='same',
            activation='relu',
            kernel_initializer=weights_initializer)(conv2d_5)

        unpool_1 = concatenate(
            [UpSampling2D(size=(2, 2))(conv2d_6), conv2d_4], axis=3)

        conv2d_7 = Conv2D(
            filters=128,
            kernel_size=(3, 3),
            strides=(1, 1),
            padding='same',
            activation='relu',
            kernel_initializer=weights_initializer)(unpool_1)

        conv2d_8 = Conv2D(
            filters=64,
            kernel_size=(3, 3),
            strides=(1, 1),
            padding='same',
            activation='relu',
            kernel_initializer=weights_initializer)(conv2d_7)

        unpool_2 = concatenate(
            [UpSampling2D(size=(2, 2))(conv2d_8), conv2d_2], axis=3)

        conv2d_9 = Conv2D(
            filters=64,
            kernel_size=(3, 3),
            strides=(1, 1),
            padding='same',
            activation='relu',
            kernel_initializer=weights_initializer)(unpool_2)
        conv2d_10 = Conv2D(
            filters=64,
            kernel_size=(3, 3),
            strides=(1, 1),
            padding='same',
            activation='relu',
            kernel_initializer=weights_initializer)(conv2d_9)

        # Conv2d_10 is 256 x 256 x 64. We now need to reduce the number of output
        # channels via a convolution with `n` filters, where `n` is the original
        # number of channels. We therefore choose `n` = 1.

        outputs = Conv2D(
            filters=1,
            kernel_size=(1, 1),
            strides=(1, 1),
            padding='same',
            activation=None,
            kernel_initializer=weights_initializer)(conv2d_10)

        optimizer = RMSprop(
            lr=LEARNING_RATE, rho=RMS_WEIGHT_DECAY, epsilon=1e-08, decay=0)

        self.model = Model(inputs=[inputs], outputs=[outputs])
        if self.num_gpus >= 2:
            self.model = multi_gpu_model(self.model, gpus=self.num_gpus)

        self.model.compile(
            optimizer=optimizer,
            loss=  FNET_ERROR_MSE, #self._parse_error(),
            metrics=[mean_squared_error])

        self.architecture_exists = True

TIME_DIR_NAME_FORMAT = "%Y%m%d-%H%M%S"


def _get_output_dir_name(suffix, exp_name=None):
    """
    Parameters
    ------------
    suffix : str
        A suffix to be appended to the directory name
    exp_name : str
        (optional) The name of the experiment. This will
        be appended to the directory name if it is not `None`

    Returns
    ------------   
    str
        The name of the output directory 
    """

    dir_name_base = datetime.now().strftime("%Y%m%d-%H%M%S")
    dir_name = "{base}-{sfx}".format(base=dir_name_base, sfx=suffix)
    if exp_name:
        dir_name = "{dn}-{en}".format(dn=dir_name, en=exp_name)

    return dir_name


def create_output_dir(base_path, suffix, exp_name=None):
    """
    Parameters
    ------------
    base_path : str
        The base directory path under which to create
        the output directory
    suffix : str
        A suffix to be appended to the directory name
    exp_name : str
        (optional) The name of the experiment. This will
        be appended to the directory name if it is not `None`

    Returns
    ------------   
    str
        The name of the created output directory
    """

    dir_name = _get_output_dir_name(suffix=suffix, exp_name=exp_name)
    dir_path = os.path.join(os.path.abspath(base_path), dir_name)

    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

    return dir_path


## Main

import numpy as np
import os
import scipy.io as sio
import re
os.environ["CUDA_VISIBLE_DEVICES"]="-1"    
import tensorflow as tf

# For Windows
file_path_fs = 'D:\Matlab_Drive\Data\MedAI\MAI_Project1\Train_image_label' # im_org - full sampled
file_path_ksps = 'D:\Matlab_Drive\Data\MedAI\MAI_Project1\Train_k'   # f_im - ksp undersampling
file_path_unds = 'D:\Matlab_Drive\Data\MedAI\MAI_Project1\Train_image_input' # im_u - undersampled
'''
# For Linux
file_path_fs = '/home/hero/Matlab_Drive/Data/MedAI/MAI_Project1/Train_image_label' # im_org - full sampled
file_path_ksps = '/home/hero/Matlab_Drive/Data/MedAI/MAI_Project1/Train_k'   # f_im - ksp undersampling
file_path_unds = '/home/hero/Matlab_Drive/Data/MedAI/MAI_Project1/Train_image_input' # im_u - undersampled
'''
# file_path = '/home/hero/Matlab_Drive/Data/ORLDB'


# Read Full Sampled Data
data_fs = {}
file_path = file_path_fs

file_list = os.listdir(file_path)
file_list = [s for s in file_list if ".mat" in s]

for file in file_list:

    os.path.join(file_path,file)
    data_read = sio.loadmat(os.path.join(file_path,file))

    data_arr = data_read['im_org']
    data_lab = re.findall('\d+',file)

    data_fs[data_lab[0]] = data_arr


# Read Under Sampled Data
data_unds = {}
file_path = file_path_unds

file_list = os.listdir(file_path)
file_list = [s for s in file_list if ".mat" in s]

for file in file_list:

    os.path.join(file_path,file)
    data_read = sio.loadmat(os.path.join(file_path,file))

    data_arr = data_read['im_u']
    data_lab = re.findall('\d+',file)

    data_unds[data_lab[0]] = data_arr


# hyper parameters

learning_rate = 0.001
training_epochs = 90
batch_size = 10
arr_size = 256 * 256


num_gpus = 1
training_error = 1

# generator
def GenMiniBatch(dict_train,batch_size):
    key_train = list(dict_train.keys())
    for i in range(batch_size):
        keys = key_train[batch_size * i : batch_size * (i+1) ]
        dict_mini = {k : dict_train[k] for k in keys}
        arr_mini = np.array(list(dict_mini.values()))
        arr_keys = np.array(keys)
        yield(arr_mini,arr_keys)

# train my model
net = FNet(num_gpus=num_gpus, error=training_error)
#for epoch in range(training_epochs):
avg_cost = 0
#total_batch = int(mnist.train.num_examples / batch_size)
total_batch = int(np.floor(len(data_unds) / batch_size))

#for i in range(total_batch):
gen = GenMiniBatch(data_unds,len(data_unds))
batch_xs, label_xs = next(gen)

dict_ys = {k : data_fs[k] for k in label_xs}
batch_ys = np.array(list(dict_ys.values()))
#c,_ = m1.train(batch_xs, batch_ys)
#avg_cost += c / total_batch
net.train(
    y_folded=batch_xs.reshape(-1,256,256,1) ,
    y_original=batch_ys.reshape(-1,256,256,1) ,
    batch_size=batch_size,
    num_epochs=training_epochs)
    #checkpoints_dir=os.getcwd() + '\\chkpnt')

#print('Epoch:', '%04d' % (epoch + 1), 'cost =', '{:.9f}'.format(avg_cost))
print('Learning Finished!')