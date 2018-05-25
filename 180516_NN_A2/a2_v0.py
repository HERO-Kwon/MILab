import numpy as np
import tensorflow as tf

##########################################
## Made by : HERO Kwon
## Title : LDA 
## Date : 2018.03.09.
## Description : LDA
##########################################

# 2018.03.25 : LDA1 finished
# 2018.03.26~ : LDA2 - cross-val, train-test split
# 2018.03.29~ : LDA3 - graphic, functionalize

# Functions
## Functiobn : LDA
def LDA_ORLDB(image_data,array_len,num_eigvec):

    print("LDA Calculation")

    img_mat = [np.array(img.reshape(1,array_len)) for img in image_data.image]
    img_mat = np.vstack(img_mat)

    # Computing Mean Vectors

    mean_vectors = []
    for cl in image_data.person.unique():
        mean_vectors.append(np.mean(img_mat[image_data.person == cl],axis=0))

    # Computing Within Scatter matrix

    S_W = np.zeros((array_len,array_len))
    for cl,mv in zip(image_data.person.unique(),mean_vectors):
        class_sc_mat = np.zeros((array_len,array_len))

        for row in img_mat[image_data.person == cl]:
            row, mv = row.reshape(array_len,1), mv.reshape(array_len,1)
            class_sc_mat += (row-mv).dot((row-mv).T)
        
        cls_prob = img_mat[image_data.person==cl].shape[0] / img_mat.shape[0]
        S_W += cls_prob * class_sc_mat

    # Computing Between scatter matrix

    overall_mean = np.mean(img_mat,axis=0)

    S_B = np.zeros((array_len,array_len))
    for mean_vec in mean_vectors:
        mean_vec = mean_vec.reshape(array_len,1)
        overall_mean = overall_mean.reshape(array_len,1)
        S_B += (mean_vec - overall_mean).dot((mean_vec - overall_mean).T)

    # Computing Eigenvalue

    eig_vals, eig_vecs = np.linalg.eig(np.linalg.pinv(S_W).dot(S_B))

    # sorting eigvectors

    eig_pairs = [(np.abs(eig_vals[i]), eig_vecs[:,i]) for i in range(len(eig_vals))]
    eig_pairs = sorted(eig_pairs,key=lambda k: k[0], reverse=True)

    #W = np.array([eig_pairs[i][1] for i in range(num_eigvec)])
    W = np.hstack([eig_pairs[i][1].reshape(array_len,1) for i in range(num_eigvec)])
    
    return(W)

## Function : Error Matrix
def Err_Mat(key_df,mat_array): 
    mat_err = np.zeros((mat_array.shape[0],mat_array.shape[0]))
    dict_err = {}

    for i in range(mat_array.shape[0]):
        for j in range(mat_array.shape[0]):
            # L2 Norm
            dist_err = np.linalg.norm(mat_array[i,:] - mat_array[j,:],2)
            mat_err[i,j] = dist_err
            if i!=j :
                key_err = ((key_df.iloc[i],key_df.iloc[j]),(i,j))
                dict_err[key_err] = dist_err

    return(mat_err,dict_err)

## Function : Distribution Curve
def Dist_Curv(tf_list,err_dict):

    print("Distribution Curve")

    dist_true = [list(err_dict.values())[i] for i in range(len(err_dict)) if tf_list[i]==True]
    dist_false = [list(err_dict.values())[i] for i in range(len(err_dict)) if tf_list[i]==False]

    plt.hist(dist_true,bins='auto',normed=1,histtype='step',color='blue',label='Dist_True')
    plt.hist(dist_false,bins='auto',normed=1,histtype='step',color='red',label='Dist_False')
    plt.legend(loc='upper right')
    plt.title("Distribution Curve")

    plt.show()

    return(dist_true,dist_false)

## Function : EER Curve
def EER_Curve(tf_list,err_dict,sampling_thres):

    print("EER Curve")

    eer_df = pd.DataFrame(columns = ['thres','fn','fp'])
    err_values = err_dict.values()
    n_thres = int(sampling_thres*len(err_values))
    sampled_err_values = random.sample(list(err_values),n_thres)

    for i, thres in enumerate(set(sampled_err_values)):
        predicted_tf = [e <= thres for e in err_dict.values()]
        
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



# Main

# packages

import numpy as np
import pandas as pd
import scipy as sp
import os
import re
import imageio
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
import random
from sklearn.metrics import confusion_matrix
import time

# import ORLDB images
# image size 56 x 46

start_time = time.time()
array_len = 56*46

# For Windows
file_path = 'D:\Matlab_Drive\Data\ORLDB'

# For Linux
# file_path = '/home/hero/Matlab_Drive/Data/ORLDB'

file_list = os.listdir(file_path)
file_list = [s for s in file_list if ".bmp" in s]

image_raw = pd.DataFrame(columns = ['image','person','person_num'])

for file in file_list:
    image_read = imageio.imread(os.path.join(file_path,file),flatten=True)
    [person,person_num] = re.findall('\d\d',file)

    data_read = {'image':image_read,'person':person,'person_num':person_num}
    image_raw = image_raw.append(data_read,ignore_index=True)

# Train-Test Split

image_train = pd.DataFrame()
image_test = pd.DataFrame()

for person in image_raw.person.unique():
    data_train, data_test = train_test_split(image_raw[image_raw.person == person],test_size = 0.5)
    image_train = image_train.append(data_train)
    image_test = image_test.append(data_test)

# Apply LDA

w_train = LDA_ORLDB(image_train,array_len,len(image_raw.person.unique())-1)

mat_train = [np.array(img.reshape(1,array_len)) for img in image_train.image]
mat_train = np.vstack(mat_train)
mat_test = [np.array(img.reshape(1,array_len)) for img in image_test.image]
mat_test = np.vstack(mat_test)

lda_train = mat_train.dot(w_train)
lda_test = mat_test.dot(w_train)

## Error Matrix
mat_err,dict_err = Err_Mat(image_test.person,lda_test)

# Distribution Curve
person_comp = [(person[0],person[1]) for person,num in list(dict_err.keys())]
person_tf = [person[0]==person[1] for person in person_comp]
dist_true,dist_false = Dist_Curv(person_tf,dict_err)

## EER Curve
eer_df = EER_Curve(person_tf,dict_err,0.1)

## Accuracy
pred_list = []
for i in range(lda_test.shape[0]):
    dist_list = []
    for j in range(lda_train.shape[0]):
        # L2 Norm
        dist_err = np.linalg.norm(lda_test[i,:] - lda_train[j,:],2)
        dist_list.append(dist_err)

    pred_person = image_train.person.iloc[np.argmin(dist_list)]
    pred_list.append(pred_person)


actual_list = list(image_train.person.values)
acc_tflist = [pred_list[i]==actual_list[i] for i in range(len(pred_list))]
acc = sum(acc_tflist) / len(acc_tflist)

print("Accuracy : " + str(acc) + " [" + str(sum(acc_tflist)) + "/" + str(len(acc_tflist)) + "]")

print("Elapsed Time : " + str(time.time() - start_time))





FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string('train_dir', '/tmp/cifar10_train',
                           """Directory where to write event logs """
                           """and checkpoint.""")
tf.app.flags.DEFINE_integer('max_steps', 390,
                            """Number of batches to run.""")
tf.app.flags.DEFINE_boolean('log_device_placement', False,
                            """Whether to log device placement.""")
tf.app.flags.DEFINE_integer('log_frequency', 10,
                            """How often to log results to the console.""")

def train():
  """Train CIFAR-10 for a number of steps."""
  with tf.Graph().as_default():
    global_step = tf.train.get_or_create_global_step()

    # Get images and labels for CIFAR-10.
    # Force input pipeline to CPU:0 to avoid operations sometimes ending up on
    # GPU and resulting in a slow down.
    with tf.device('/cpu:0'):
      images, labels = inputs(eval_data) #cifar10.distorted_inputs()

    # Build a Graph that computes the logits predictions from the
    # inference model.
    logits = inference(images)

    # Calculate loss.
    loss = loss(logits, labels)
    tf.summary.scalar('loss', loss)
    # Build a Graph that trains the model with one batch of examples and
    # updates the model parameters.
    train_op = train(loss, global_step)

    class _LoggerHook(tf.train.SessionRunHook):
      """Logs loss and runtime."""

      def begin(self):
        self._step = -1
        self._start_time = time.time()

      def before_run(self, run_context):
        self._step += 1
        return tf.train.SessionRunArgs(loss)  # Asks for loss value.

      def after_run(self, run_context, run_values):
        if self._step % FLAGS.log_frequency == 0:
          current_time = time.time()
          duration = current_time - self._start_time
          self._start_time = current_time

          loss_value = run_values.results
          examples_per_sec = FLAGS.log_frequency * FLAGS.batch_size / duration
          sec_per_batch = float(duration / FLAGS.log_frequency)

          format_str = ('%s: step %d, loss = %.2f (%.1f examples/sec; %.3f '
                        'sec/batch)')
          print (format_str % (datetime.now(), self._step, loss_value,
                               examples_per_sec, sec_per_batch))
          cifar10_eval.evaluate()

    with tf.train.MonitoredTrainingSession(
        checkpoint_dir=FLAGS.train_dir,
        save_summaries_steps = 10,
        save_checkpoint_secs = 1,
        hooks=[tf.train.StopAtStepHook(last_step=FLAGS.max_steps),
               tf.train.NanTensorHook(loss),
               _LoggerHook()],
        config=tf.ConfigProto(
            log_device_placement=FLAGS.log_device_placement)) as mon_sess:
      while not mon_sess.should_stop():
        mon_sess.run(train_op)


def main(argv=None):  # pylint: disable=unused-argument
  cifar10.maybe_download_and_extract()
  if tf.gfile.Exists(FLAGS.train_dir):
    tf.gfile.DeleteRecursively(FLAGS.train_dir)
  tf.gfile.MakeDirs(FLAGS.train_dir)
  train()


if __name__ == '__main__':
  tf.app.run()