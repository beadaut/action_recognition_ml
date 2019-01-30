import glob
import cv2
import copy
import random
import time

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

from copy import deepcopy
from scipy import ndimage
from sklearn.utils import shuffle
from mpl_toolkits.mplot3d import Axes3D

from utils.config import cfg
from utils.data_from_vid_generator import read_clip_to_frame_bundles, load_txt_filenames
from chalearn_motion_model import build_graph


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
      self.keep_prob_pl = tf.placeholder(tf.float32)

      self.pred = build_graph(self.inputs_pl, self.is_training_pl, keep_prob=self.keep_prob_pl,
                                 weight_decay=0.0, bn_decay=None)

      # print("logits loaded...")
      # self.pred=tf.nn.softmax(pred)

      self.sess = tf.Session()
      saver = tf.train.Saver()
      saver.restore(self.sess, model_path)
      print("\nLoaded model... ", model_path)
  

def training_inference():
  model_path = 'logdir_chalearn_lr_simple_ff_10_96/model_epoch_120'

  filenames = np.load(
      '/media/tjosh/vault/chalearn_dataset/IsoGR_2016/train_validation_filenames_2.npy')
  labels = np.load(
      '/media/tjosh/vault/chalearn_dataset/IsoGR_2016/train_validation_labels_2.npy')
  
  inference_model = InferenceModel(num_frames=cfg.num_frames, model_path=model_path)
  
  data_size = len(filenames)
  correct_count = 0
  for i in range(data_size):
    print("frame: {} of {}".format(i, data_size))
    current_clip = "/media/tjosh/vault/chalearn_dataset/IsoGR_2016/train/"+filenames[i]
    current_label = int(labels[i])
    # print(current_label)
    if current_label==249:
      current_label=0
    clip_bundles = read_clip_to_frame_bundles(current_clip, cfg.num_frames)
    # print(len(clip_bundles))
    current_bundle_size = len(clip_bundles)

    predictions_array = np.zeros(249)

    for frame_i in range(current_bundle_size):
      data_in = clip_bundles[frame_i]
      data_in = np.transpose(np.expand_dims(np.array(data_in), 0), (0, 2, 3, 1))
      # print(np.shape(data_in))
      
      # do inference here:
      prediction = inference_model.sess.run(
          inference_model.pred,
          feed_dict={inference_model.inputs_pl: data_in,
                    inference_model.keep_prob_pl: 1.0,
                    inference_model.is_training_pl: False})
      
      predictions_array += prediction[0]
    prediction_final = np.argmax(predictions_array)
    correct = int(current_label == prediction_final)
    correct_count += int(correct)
    print("Label: {}, Prediction: {}".format(current_label, prediction_final))
  
  print("Final Accuracy: ", correct_count/data_size)


def test_inference():
  """
  do inference and write results to file for submission
  """
  data_tag = "test"
  model_path = 'logdir_chalearn_lr_simple_ff_10_96/model_epoch_140'
  m_filenames, k_filenames = load_txt_filenames(
      "/media/tjosh/vault/chalearn_dataset/IsoGR_2016/"+data_tag+"/"+data_tag+"_list.txt")

  inference_model = InferenceModel(
      num_frames=cfg.num_frames, model_path=model_path)

  data_size = len(k_filenames)

  file = open(data_tag+"_list_results.txt", "w")


  for i in range(data_size):
    print("frame: {} of {}".format(i, data_size))
    current_clip = "/media/tjosh/vault/chalearn_dataset/IsoGR_2016/"+data_tag+"/" + \
        k_filenames[i]

    # print(current_clip)
    
    
    clip_bundles = read_clip_to_frame_bundles(current_clip, cfg.num_frames)
    # print(len(clip_bundles))
    current_bundle_size = len(clip_bundles)

    predictions_array = np.zeros(249)

    for frame_i in range(current_bundle_size):
      data_in = clip_bundles[frame_i]
      data_in = np.transpose(np.expand_dims(
          np.array(data_in), 0), (0, 2, 3, 1))
      # print(np.shape(data_in))

      # do inference here:
      prediction = inference_model.sess.run(
          inference_model.pred,
          feed_dict={inference_model.inputs_pl: data_in,
                     inference_model.keep_prob_pl: 1.0,
                     inference_model.is_training_pl: False})

      predictions_array += prediction[0]
    prediction_final = np.argmax(predictions_array)
    if prediction_final == 0:
      prediction_final = 249
    
    print("Prediction: ", prediction_final)
    file.write(m_filenames[i]+" "+k_filenames[i] +
               " "+str(prediction_final)+"\n")
    
    # if i > 10:
    #   break
  
  file.close()


  

if __name__ == '__main__':
  test_inference()
  # m_filenames, k_filenames = load_txt_filenames(
  #     "/media/tjosh/vault/chalearn_dataset/IsoGR_2016/test/test_list.txt")
  # data_size = len(m_filenames)
  
  # file = open("testfile.txt", "w")

  # for i in range(data_size):
  #   file.write(m_filenames[i]+" "+k_filenames[i]+" label\n")
  #   # print(m_filenames[i], " ", k_filenames[i])
  
  # file.close()
