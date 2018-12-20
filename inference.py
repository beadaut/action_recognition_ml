import glob
import csv
import cv2
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

      self.pred = build_graph(self.inputs_pl, self.is_training_pl,
                        weight_decay=0.0, bn_decay=None)

      # config_ss = tf.ConfigProto()
      self.sess = tf.Session()
      saver = tf.train.Saver()
      saver.restore(self.sess, model_path)
      print("\nLoaded model... ", model_path)


def do_inference(filename, inference_model, time_steps=5, display_images=False):
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
    # print(csv_reader)
    all_data = []
    for i, row in enumerate(csv_reader):
      # row_vals = row[0].split(',')
      # print(len(row_vals))
      all_data+=row
      # print(row)
      # print(i)

    all_data = [int(x) for x in all_data]
    # print(len(all_data))

  all_data = np.reshape(all_data, [240*320,-1])
  # all_data = np.reshape(all_data, [-1,240,320])

  # print(np.shape(all_data))
  # print(np.shape(all_data[:,0]))
  # print(all_data[0])


  # with tf.Graph().as_default():
  #   inputs_pl = tf.placeholder(
  #       tf.float32, shape=(
  #           1, 240, 320, cfg.num_frames))
  #   is_training_pl = tf.placeholder(tf.bool, shape=())

  #   pred = build_graph(inputs_pl, is_training_pl, weight_decay=0.0, bn_decay=None)

  #   # config_ss = tf.ConfigProto()
  #   sess = tf.Session()
  #   saver = tf.train.Saver()
  #   saver.restore(sess, model_path)
  #   print("\nLoaded model... ", model_path)

  predictions_array = np.zeros(20)
  for i in range(np.shape(all_data)[-1]):
    if i < time_steps:
      continue
    input_bundle = [np.reshape(all_data[:,i-x], [240,320]) for x in range(time_steps)]

    # do prediction here:
    data_in = [np.transpose(input_bundle, (1, 2, 0))]
    # print("shape of input data: ", np.shape(data_in))
    prediction = inference_model.sess.run(
        inference_model.pred, 
        feed_dict={inference_model.inputs_pl: data_in, 
                    inference_model.is_training_pl: False})
    
    predictions_array += prediction[0]
    
    predict_class = np.argmax(prediction[0])
    
    # print("prediction: ", prediction)
    # print("predict class now: ", predict_class)


    if display_images:
      cols, rows = [240,320]
      image = np.reshape(all_data[:,i], [cols, rows])

      # test transformation (augmentation)
      # transf_x = int(random.random()*100) - 50
      # M = np.float32([[1,0,0],[0,1,-50]])
      # image = ndimage.affine_transform(image, M)
      cv2.imshow("Lidar Camera", image)
      key = cv2.waitKey(100)
      if key == 27:   # Esc key
          break

  # print('Shape of input bundle: ', np.shape(input_bundle))
  # label = filename.split('/')[1].split('_')[0].split('a')[1]
  label = int(filename.split('/')[-1].split('_')[0].split('a')[1])
  if label==20:
    label=0
  # print("Label: ", label)
  # labels_count[int(label)-1]+=1
  # print('Lable: ', int(label))
  # print('Label count: ', labels_count)
  prediction_final = np.argmax(predictions_array)
  # print("Final prediction: ", prediction_final)
  print("Label: {}, Prediction: {}".format(label, prediction_final))
  correct = int(label==prediction_final)
  # print("Correct: ", correct)

  return correct


# test:
# filename = '/home/tjosh/datasets/MSRAction3D/csv/a18_s02_e02_sdepth.csv'
# correct = do_inference(filename, model_path, time_steps=5)
# set_1_labels = ['02', '03', '05', '06', '10', '13', '18', '20']
# set_2_labels = ['01', '04', '07', '08', '09', '11', '12', '14']
# set_3_labels = ['06', '14', '15', '16', '17', '18', '19', '20']

set_labels = ['06', '14', '15', '16', '17', '18', '19', '20']
model_path = 'logdir_set_3_t2_simple_ff_5_96/model_epoch_100'

all_samples = []
for label in set_labels:
  datasamples = glob.glob('csv/a'+label+'_s*[01-10]_e0*[0-9]_sdepth.csv')
  # datasamples = glob.glob('/home/tjosh/datasets/MSRAction3D/csv/*.csv')
  all_samples += datasamples
  # for sample in datasamples:
  #   print(sample)

samples_size = len(all_samples)
print("Samples Size: ", samples_size)
time_steps = 5
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
    correct_count += 1
    pass
  
print("Correct counts: ", correct_count)
print("Final accuracy: ", correct_count/float(samples_size))

# set_2_labels = [1, 4, 7, 8, 9, 11, 12, 14]
# set_3_labels = [6, 14, 15, 16, 17, 18, 19, 20]


# all_files = glob.glob('csv/*.csv')

# for i, filename in enumerate(all_files):
#   # if i<388:
#   #   continue
#   try:
#     # pass
#     print('\n%d of %d'%(i, len(all_files)))
#     csv2input_data(filename, time_steps=4, save_dir='/media/tjosh/vault/MSRAction3D/npy_4')
#   except Exception as e:
#     print('Error: ',e)
#     continue

# # after everything
# print('Number of files processed: ', len(all_files))


# # cut make training set and validation set
# all_data = glob.glob('/media/tjosh/vault/MSRAction3D/npy_5/*.npy')
# print(len(all_data))
# size_of_data = len(all_data)
# all_data = shuffle(all_data)
# training_cut = all_data[int(0.2*size_of_data):]
# validation_cut = all_data[:int(0.2*size_of_data)]
# np.save('/media/tjosh/vault/MSRAction3D/npy_5_training', training_cut)
# np.save('/media/tjosh/vault/MSRAction3D/npy_5_validation', validation_cut)
# print('all saved!')
