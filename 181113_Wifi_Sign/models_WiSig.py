def CNN1d_1(in_data):
    input1d = tf.reshape(in_data, [-1,max_len,6])
    conv1 = tf.layers.conv1d(
        inputs=input1d, 
        filters=100, 
        kernel_size=3, 
        strides=1,
        #padding="same", 
        activation=tf.nn.relu)
    pool1 = tf.layers.max_pooling1d(
        inputs=conv1, 
        pool_size=10, 
        strides=10)
        #padding="same")
    conv2 = tf.layers.conv1d(
        inputs=pool1, 
        filters=250, 
        kernel_size=3, 
        strides=1,
        #padding="same", 
        activation=tf.nn.relu)
    pool2 = tf.layers.max_pooling1d(
        inputs=conv2, 
        pool_size=10, 
        strides=10)
        #padding="same")
    pool_flat = tf.layers.flatten(pool2)
    fc = tf.layers.dense(
        inputs= pool_flat, units=500, activation=tf.nn.relu)
    output = tf.layers.dense(inputs=fc, units=max_id+1)    
    return output


def CNN1d_2(in_data):
    input1d = tf.reshape(in_data, [-1,max_len,6])
    conv11 = tf.layers.conv1d(
        inputs=input1d, 
        filters=32, 
        kernel_size=3, 
        strides=1,
        padding="same", 
        activation=tf.nn.relu)
    conv12 = tf.layers.conv1d(
        inputs=conv11, 
        filters=32, 
        kernel_size=3, 
        strides=1,
        padding="same", 
        activation=tf.nn.relu)
    pool1 = tf.layers.max_pooling1d(
        inputs=conv12, 
        pool_size=10, 
        strides=10)
    conv21 = tf.layers.conv1d(
        inputs=pool1, 
        filters=64, 
        kernel_size=3, 
        strides=1,
        padding="same", 
        activation=tf.nn.relu)
    conv211 = tf.layers.conv1d(
        inputs=conv21, 
        filters=32, 
        kernel_size=1, 
        strides=1,
        padding="same", 
        activation=None)
    conv22 = tf.layers.conv1d(
        inputs=conv211, 
        filters=64, 
        kernel_size=3, 
        strides=1,
        padding="same", 
        activation=tf.nn.relu)
    
    pool2 = tf.layers.max_pooling1d(
        inputs=conv22, 
        pool_size=10,
        strides=10)
    
    conv31 = tf.layers.conv1d(
        inputs=pool2, 
        filters=128, 
        kernel_size=3, 
        strides=1,
        padding="same", 
        activation=tf.nn.relu)
    conv311 = tf.layers.conv1d(
        inputs=conv31, 
        filters=64, 
        kernel_size=1, 
        strides=1,
        padding="same", 
        activation=tf.nn.relu)
    conv32 = tf.layers.conv1d(
        inputs=conv311, 
        filters=128, 
        kernel_size=3, 
        strides=1,
        padding="same", 
        activation=tf.nn.relu)
    conv321 = tf.layers.conv1d(
        inputs=conv32, 
        filters=64, 
        kernel_size=1, 
        strides=1,
        padding="same", 
        activation=tf.nn.relu)
    conv33 = tf.layers.conv1d(
        inputs=conv321, 
        filters=128, 
        kernel_size=3, 
        strides=1,
        padding="same", 
        activation=tf.nn.relu)
    
    pool3 = tf.layers.max_pooling1d(
        inputs=conv33, 
        pool_size=10, 
        strides=10)
    
    pool_flat = tf.layers.flatten(pool3)
    fc = tf.layers.dense(
        inputs= pool_flat, units=500)#, activation=tf.nn.relu)
    #fc_drop = tf.nn.dropout(fc, keep_prob) 
    output = tf.layers.dense(inputs=fc, units=max_id+1) 
    return output