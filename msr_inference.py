import glob
import csv
# import cv2
import random
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

import time
import tensorflow as tf

from utils.config import cfg

from scipy import ndimage
from sklearn.utils import shuffle

from motion_model import build_graph, get_loss
from convert_depth2pc import generate_pointcloud

labels_count = [0]*20


class InferenceModel(object):
  """docstring for InferenceModel"""

  def __init__(self, num_frames, model_path):
    super(InferenceModel, self).__init__()
    self.num_frames = num_frames
    self.model_path = model_path

    with tf.Graph().as_default():
      self.inputs_pl = tf.placeholder(
          tf.float32, shape=(
              1, 240, 320, cfg.num_frames))
      self.is_training_pl = tf.placeholder(tf.bool, shape=())

      pred = build_graph(self.inputs_pl, self.is_training_pl,
                        weight_decay=0.0, bn_decay=None)

      self.pred=tf.nn.softmax(pred)

      self.sess = tf.Session()
      saver = tf.train.Saver()
      saver.restore(self.sess, model_path)
      print("\nLoaded model... ", model_path)


def do_inference(filename, inference_model, time_steps=5, display_images=False, pc_inputs=False):
  """
  open motion csv data and create input data for training model
  params:
    filename: path to cvs file
    time_steps: number of time steps to bundle
    display_image: wether to show images while processing. Default is False
    save_dir: Directory to save processed data. Default is 'npy'
  """
  with open(filename) as csv_file:
    csv_reader = csv.reader(csv_file) 
    
    all_data = []
    for i, row in enumerate(csv_reader):
      
      all_data+=row

    all_data = [int(x) for x in all_data]

  all_data = np.reshape(all_data, [240*320,-1])


  # predictions_array = np.zeros(20)
  predictions_array = np.ones(20)
  for i in range(np.shape(all_data)[-1]):
    if i < time_steps:
      continue
    # input_bundle = [np.reshape(all_data[:,i-x], [240,320]) for x in range(time_steps)] # old solution
    
    input_bundle = []
    for x in range(time_steps):
      input_i_x = np.reshape(all_data[:, i-x], [240, 320])
      input_bundle.append(input_i_x)

    # do prediction here:
    data_in = [np.transpose(input_bundle, (1, 2, 0))]

    if pc_inputs:
          data_in = generate_pointcloud(data_in)  # , max_points=1024
    
    prediction = inference_model.sess.run(
        inference_model.pred, 
        feed_dict={inference_model.inputs_pl: data_in, 
                    inference_model.is_training_pl: False})
    
    predictions_array *= prediction[0]
    
    # predict_class = np.argmax(prediction[0])
    # print("predict class now: ", predict_class)
    
    if display_images:
      cols, rows = [240,320]
      image = np.reshape(all_data[:,i], [cols, rows])

      cv2.imshow("Lidar Camera", image)
      key = cv2.waitKey(100)
      if key == 27:   # Esc key
          break

  label = int(filename.split('/')[-1].split('_')[0].split('a')[1])
  if label==20:
    label=0

  prediction_final = np.argmax(predictions_array)
  correct = int(label==prediction_final)
  
  print("Label: {}, Prediction: {}".format(label, prediction_final))

  return correct


# model_path = 'logdir_all_msr_t3_aug_simple_ff_5_96/model_epoch_60'
# model_path = 'logdir_new_all_msr_t3_aug_simple_ff_5_96/model_epoch_5'
model_path = '/media/tjosh/vault/MSRAction3D/trained_models/logdir_all_msr_t3_re_simple_ff_5_96/model_epoch_40'

# set_1_labels = ['02', '03', '05', '06', '10', '13', '18', '20']
# set_2_labels = ['01', '04', '07', '08', '09', '11', '12', '14']
# set_3_labels = ['06', '14', '15', '16', '17', '18', '19', '20']
# all_labels = ['01', '02', '03', '04', '05', '06', '07', '08', '09', '10', '11', '12', '13', '14', '15', '16', '17', '18', '19', '20']


set_labels = ['01', '02', '03', '04', '05', '06', '07', '08', '09',
              '10', '11', '12', '13', '14', '15', '16', '17', '18', '19', '20']

all_samples = []
for label in set_labels:
  datasamples = glob.glob(
      '/media/tjosh/vault/MSRAction3D/csv/a'+label+'_s*[0-9]_e*[0-9]_sdepth.csv')
  for samples in datasamples:
    all_samples.append(samples)

samples_size = len(all_samples)
print("Samples Size: ", samples_size)
time_steps = cfg.num_frames
inference_model = InferenceModel(num_frames=time_steps, model_path=model_path)

correct_count = 0
for i, sample in enumerate(all_samples):
  print("sample: {}/{}".format(i+1, len(all_samples)))
  try:
    correct = do_inference(sample, inference_model, time_steps=time_steps)
    correct_count += correct
    if correct ==0:
      print("***")
  except Exception as identifier:
    print(identifier)
    correct_count += 1
    pass
  
print("Correct counts: ", correct_count)
print("Final accuracy: ", correct_count/float(samples_size))

