### Title : AUTOMAP v1
### Made by : HERO Kwon
### Date : 20180425
### Referenced : https://github.com/tetianadadakova/MRI-CNN


import numpy as np
import os
import scipy.io as sio
import re
import tensorflow as tf
from tensorflow.python.framework import ops
import math
import time
os.environ["CUDA_VISIBLE_DEVICES"]="-1"    

# For Windows
file_path_fs = 'D:\Matlab_Drive\Data\MedAI\MAI_Project1\Train_image_label' # im_org - full sampled
file_path_ksps = 'D:\Matlab_Drive\Data\MedAI\MAI_Project1\Train_k'   # f_im - ksp undersampling
file_path_unds = 'D:\Matlab_Drive\Data\MedAI\MAI_Project1\Train_image_input' # im_u - undersampled



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
training_epochs = 15
batch_size = 2
arr_size = 256 * 256


#from generate_input import load_images_from_folder


def load_images_from_folder(folder, n_im, normalize=False, imrotate=False):
    """ Loads n_im images from the folder and puts them in an array bigy of
    size (n_im, im_size1, im_size2), where (im_size1, im_size2) is an image
    size.
    Performs FFT of every input image and puts it in an array bigx of size
    (n_im, im_size1, im_size2, 2), where "2" represents real and imaginary
    dimensions
    :param folder: path to the folder, which contains images
    :param n_im: number of images to load from the folder
    :param normalize: if True - the xbig data will be normalized
    :param imrotate: if True - the each input image will be rotated by 90, 180,
    and 270 degrees
    :return:
    bigx: 4D array of frequency data of size (n_im, im_size1, im_size2, 2)
    bigy: 3D array of images of size (n_im, im_size1, im_size2)
    """

    # Initialize the arrays:
    if imrotate:  # number of images is 4 * n_im
        bigy = np.empty((n_im * 4, 256, 256))
        bigx = np.empty((n_im * 4, 256, 256, 2))
    else:
        bigy = np.empty((n_im, 256, 256))
        bigx = np.empty((n_im, 256, 256, 2))

    im = 0  # image counter
    for filename in os.listdir(folder):
        if not filename.startswith('.'):
            #bigy_temp = cv2.imread(os.path.join(folder, filename),
            #                       cv2.IMREAD_GRAYSCALE)

            bigy_temp = sio.loadmat(os.path.join(folder,filename))['im_u']
            bigy[im, :, :] = bigy_temp
            bigx[im, :, :, :] = create_x(bigy_temp, normalize)
            im += 1
            if imrotate:
                for angle in [90, 180, 270]:
                    bigy_rot = im_rotate(bigy_temp, angle)
                    bigx_rot = create_x(bigy_rot, normalize)
                    bigy[im, :, :] = bigy_rot
                    bigx[im, :, :, :] = bigx_rot
                    im += 1

        if imrotate:
            if im > (n_im * 4 - 1):  # how many images to load
                break
        else:
            if im > (n_im - 1):  # how many images to load
                break

    if normalize:
        bigx = (bigx - np.amin(bigx)) / (np.amax(bigx) - np.amin(bigx))

    return bigx, bigy


def create_x(y, normalize=False):
    """
    Prepares frequency data from image data: applies to_freq_space,
    expands the dimensions from 3D to 4D, and normalizes if normalize=True
    :param y: input image
    :param normalize: if True - the frequency data will be normalized
    :return: frequency data 4D array of size (1, im_size1, im_size2, 2)
    """
    x = to_freq_space(y)  # FFT: (128, 128, 2)
    x = np.expand_dims(x, axis=0)  # (1, 128, 128, 2)
    if normalize:
        x = x - np.mean(x)

    return x


def to_freq_space(img):
    """ Performs FFT of an image
    :param img: input 2D image
    :return: Frequency-space data of the input image, third dimension (size: 2)
    contains real ans imaginary part
    """

    img_f = np.fft.fft2(img)  # FFT
    img_fshift = np.fft.fftshift(img_f)  # FFT shift
    img_real = img_fshift.real  # Real part: (im_size1, im_size2)
    img_imag = img_fshift.imag  # Imaginary part: (im_size1, im_size2)
    img_real_imag = np.dstack((img_real, img_imag))  # (im_size1, im_size2, 2)

    return img_real_imag


def im_rotate(img, angle):
    """ Rotates an image by angle degrees
    :param img: input image
    :param angle: angle by which the image is rotated, in degrees
    :return: rotated image
    """
    rows, cols = img.shape
    rotM = cv2.getRotationMatrix2D((cols/2-0.5, rows/2-0.5), angle, 1)
    imrotated = cv2.warpAffine(img, rotM, (cols, rows))

    return imrotated


def create_placeholders(n_H0, n_W0):
    """ Creates placeholders for x and y for tf.session
    :param n_H0: image height
    :param n_W0: image width
    :return: x and y - tf placeholders
    """

    x = tf.placeholder(tf.float16, shape=[None, n_H0, n_W0, 2], name='x')
    y = tf.placeholder(tf.float16, shape=[None, n_H0, n_W0], name='y')

    return x, y


def initialize_parameters():
    """ Initializes filters for the convolutional and de-convolutional layers
    :return: parameters - a dictionary of filters (W1 - first convolutional
    layer, W2 - second convolutional layer, W3 - de-convolutional layer
    """

    W1 = tf.get_variable("W1", [25, 25, 1, 64],  # 64 filters of size 5x5
                         dtype=tf.float16,
                         initializer=tf.contrib.layers.xavier_initializer
                         (seed=0))
    W2 = tf.get_variable("W2", [25, 25, 64, 64],  # 64 filters of size 5x5
                         dtype=tf.float16,
                         initializer=tf.contrib.layers.xavier_initializer
                         (seed=0))
    W3 = tf.get_variable("W3", [49, 49, 1, 64],  # 64 filters of size 7x7
                         dtype=tf.float16,
                         initializer=tf.contrib.layers.xavier_initializer
                         (seed=0))  # conv2d_transpose

    parameters = {"W1": W1,
                  "W2": W2,
                  "W3": W3}

    return parameters


def forward_propagation(x, parameters):
    """ Defines all layers for forward propagation:
    Fully connected (FC1) -> tanh activation: size (n_im, n_H0 * n_W0)
    -> Fully connected (FC2) -> tanh activation:  size (n_im, n_H0 * n_W0)
    -> Convolutional -> ReLU activation: size (n_im, n_H0, n_W0, 64)
    -> Convolutional -> ReLU activation with l1 regularization: size (n_im, n_H0, n_W0, 64)
    -> De-convolutional: size (n_im, n_H0, n_W0)
    :param x: Input - images in frequency space, size (n_im, n_H0, n_W0, 2)
    :param parameters: parameters of the layers (e.g. filters)
    :return: output of the last layer of the neural network
    """

    x_temp = tf.reshape(x,[-1,x.shape[1] * x.shape[2] *2]),#tf.contrib.layers.flatten(x)  # size (n_im, n_H0 * n_W0 * 2)
    n_out = np.int(x.shape[1] * x.shape[2])  # size (n_im, n_H0 * n_W0)

    #with tf.device('/cpu:0'):
        # FC: input size (n_im, n_H0 * n_W0 * 2), output size (n_im, n_H0 * n_W0)
    FC1 = tf.contrib.layers.fully_connected(
        x_temp,
        n_out,
        activation_fn=tf.tanh,
        normalizer_fn=None,
        normalizer_params=None,
        weights_initializer=tf.contrib.layers.xavier_initializer(),
        weights_regularizer=None,
        biases_initializer=None,
        biases_regularizer=None,
        reuse=tf.AUTO_REUSE,
        variables_collections=None,
        outputs_collections=None,
        trainable=True,
        scope='fc1')

    #with tf.device('/cpu:0'):
        # FC: input size (n_im, n_H0 * n_W0), output size (n_im, n_H0 * n_W0)
    FC2 = tf.contrib.layers.fully_connected(
        FC1,
        n_out,
        activation_fn=tf.tanh,
        normalizer_fn=None,
        normalizer_params=None,
        weights_initializer=tf.contrib.layers.xavier_initializer(),
        weights_regularizer=None,
        biases_initializer=None,
        biases_regularizer=None,
        reuse=tf.AUTO_REUSE,
        variables_collections=None,
        outputs_collections=None,
        trainable=True,
        scope='fc2')
    
    # Reshape output from FC layers into array of size (n_im, n_H0, n_W0, 1):
    #FC_M = tf.reshape(FC2, [tf.shape(x)[0], tf.shape(x)[1], tf.shape(x)[2], 1])
    FC_M = tf.reshape(FC2, [tf.shape(x)[0], tf.shape(x)[1], tf.shape(x)[2], 1])

    # Retrieve the parameters from the dictionary "parameters":
    W1 = parameters['W1']
    W2 = parameters['W2']
    W3 = parameters['W3']

    # CONV2D: filters W1, stride of 1, padding 'SAME'
    # Input size (n_im, n_H0, n_W0, 1), output size (n_im, n_H0, n_W0, 64)
    Z1 = tf.nn.conv2d(FC_M, W1, strides=[1, 1, 1, 1], padding='SAME')
    # RELU
    CONV1 = tf.nn.relu(Z1)

    # CONV2D: filters W2, stride 1, padding 'SAME'
    # Input size (n_im, n_H0, n_W0, 64), output size (n_im, n_H0, n_W0, 64)
    #Z2 = tf.nn.conv2d(CONV1, W2, strides=[1, 1, 1, 1], padding='SAME')
    # RELU
    #CONV2 = tf.nn.relu(Z2)
    
    CONV2 = tf.layers.conv2d(
        CONV1,
        filters=64,
        kernel_size=5,
        strides=(1, 1),
        padding='same',
        data_format='channels_last',
        dilation_rate=(1, 1),
        activation=tf.nn.relu,
        use_bias=True,
        kernel_initializer=None,
        bias_initializer=tf.zeros_initializer(),
        kernel_regularizer=tf.contrib.layers.l1_regularizer(0.0001),
        bias_regularizer=None,
        activity_regularizer=None,
        kernel_constraint=None,
        bias_constraint=None,
        trainable=True,
        name='conv2',
        reuse=tf.AUTO_REUSE)
    
    # DE-CONV2D: filters W3, stride 1, padding 'SAME'
    # Input size (n_im, n_H0, n_W0, 64), output size (n_im, n_H0, n_W0, 1)
    batch_size = tf.shape(x)[0]
    deconv_shape = tf.stack([batch_size, x.shape[1], x.shape[2], 1])
    DECONV = tf.nn.conv2d_transpose(CONV2, W3, output_shape=deconv_shape,
                                    strides=[1, 1, 1, 1], padding='SAME')
    DECONV = tf.squeeze(DECONV)

    return DECONV


def compute_cost(DECONV, Y):
    """
    Computes cost (squared loss) between the output of forward propagation and
    the label image
    :param DECONV: output of forward propagation
    :param Y: label image
    :return: cost (squared loss)
    """

    cost = tf.square(DECONV - Y)

    return cost


def random_mini_batches(x, y, mini_batch_size=64, seed=0):
    """ Shuffles training examples and partitions them into mini-batches
    to speed up the gradient descent
    :param x: input frequency space data
    :param y: input image space data
    :param mini_batch_size: mini-batch size
    :param seed: can be chosen to keep the random choice consistent
    :return: a mini-batch of size mini_batch_size of training examples
    """

    m = x.shape[0]  # number of input images
    mini_batches = []
    np.random.seed(seed)

    # Shuffle (x, y)
    permutation = list(np.random.permutation(m))
    shuffled_X = x[permutation, :]
    shuffled_Y = y[permutation, :]

    # Partition (shuffled_X, shuffled_Y). Minus the end case.
    num_complete_minibatches = int(math.floor(
        m / mini_batch_size))  # number of mini batches of size mini_batch_size

    for k in range(0, num_complete_minibatches):
        mini_batch_X = shuffled_X[k * mini_batch_size:k * mini_batch_size
                                    + mini_batch_size, :, :, :]
        mini_batch_Y = shuffled_Y[k * mini_batch_size:k * mini_batch_size
                                    + mini_batch_size, :, :]
        mini_batch = (mini_batch_X, mini_batch_Y)
        mini_batches.append(mini_batch)

    # Handling the end case (last mini-batch < mini_batch_size)
    if m % mini_batch_size != 0:
        mini_batch_X = shuffled_X[num_complete_minibatches
                                  * mini_batch_size: m, :, :, :]
        mini_batch_Y = shuffled_Y[num_complete_minibatches
                                  * mini_batch_size: m, :, :]
        mini_batch = (mini_batch_X, mini_batch_Y)
        mini_batches.append(mini_batch)

    return mini_batches


def model(X_train, Y_train, learning_rate=0.0001,
          num_epochs=100, minibatch_size=64, print_cost=True):
    """ Runs the forward and backward propagation
    :param X_train: input training frequency-space data
    :param Y_train: input training image-space data
    :param learning_rate: learning rate of gradient descent
    :param num_epochs: number of epochs
    :param minibatch_size: size of mini-batch
    :param print_cost: if True - the cost will be printed every epoch, as well
    as how long it took to run the epoch
    :return: this function saves the model to a file. The model can then
    be used to reconstruct the image from frequency space
    """   

    #with tf.device('/cpu:0'):
    ops.reset_default_graph()  # to not overwrite tf variables
    seed = 3
    (m, n_H0, n_W0,_) = X_train.shape

    # Create Placeholders
    X, Y = create_placeholders(n_H0, n_W0)

    # Initialize parameters
    parameters = initialize_parameters()

    # Build the forward propagation in the tf graph
    DECONV = forward_propagation(X, parameters)

    # Add cost function to tf graph
    cost = compute_cost(DECONV, Y)

    # Backpropagation
    optimizer = tf.train.RMSPropOptimizer(learning_rate).minimize(cost)

    # Initialize all the variables globally
    init = tf.global_variables_initializer()

    # Add ops to save and restore all the variables
    saver = tf.train.Saver()

    # For memory
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True

    # Memory config
    #config = tf.ConfigProto()
    #config.gpu_options.allow_growth = True
    config = tf.ConfigProto(log_device_placement=True)

    # Start the session to compute the tf graph
    with tf.Session(config=config) as sess:

        # Initialization
        sess.run(init)

        # Training loop
        for epoch in range(num_epochs):
            tic = time.time()

            minibatch_cost = 0.
            num_minibatches = int(m / minibatch_size)  # number of minibatches
            seed += 1
            minibatches = random_mini_batches(X_train, Y_train,
                                                minibatch_size, seed)
            # Minibatch loop
            for minibatch in minibatches:
                # Select a minibatch
                (minibatch_X, minibatch_Y) = minibatch
                # Run the session to execute the optimizer and the cost
                _, temp_cost = sess.run(
                    [optimizer, cost],
                    feed_dict={X: minibatch_X, Y: minibatch_Y})

                cost_mean = np.mean(temp_cost) / num_minibatches
                minibatch_cost += cost_mean

            # Print the cost every epoch
            if print_cost:
                toc = time.time()
                print ('EPOCH = ', epoch, 'COST = ', minibatch_cost, 'Elapsed time = ', (toc - tic))

        # Save the variables to disk.
        save_path = saver.save(sess, "model_auto/model_name.ckpt")
        print("Model saved in file: %s" % save_path)

        sess.close()



# Load training data:

tic1 = time.time()
dir_train = 'D:\Matlab_Drive\Data\MedAI\MAI_Project1\Train_image_input'  # Folder with images
n_im = 20  # How many images to load
X_train, Y_train = load_images_from_folder(  # Load images for training
    dir_train,
    n_im,
    normalize=False,
    imrotate=False)
toc1 = time.time()

print('Time to load data = ', (toc1 - tic1))
print('X_train.shape at input = ', X_train.shape)
print('Y_train.shape at input = ', Y_train.shape)



# Finally run the model!

model(np.float32(X_train[:,:64,:64,:]), np.float32(Y_train[:,:64,:64]),
      learning_rate=0.00002,
      num_epochs=10,
      minibatch_size=5,  # should be < than the number of input examples
      print_cost=True)