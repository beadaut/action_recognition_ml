import os
import re
import csv
# import cv2
import glob
import json
import time
import random
import xml.etree.ElementTree

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import tensorflow as tf

from scipy import ndimage
from sklearn.utils import shuffle

from utils.config import cfg
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

      pred = build_graph(self.inputs_pl, self.is_training_pl,
                        weight_decay=0.0, bn_decay=None)

      self.pred=tf.nn.softmax(pred)

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


  # predictions_array = np.zeros(20)
  predictions_array = np.ones(cfg.num_class)
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
    
    predictions_array *= prediction[0]
    
    predict_class = np.argmax(prediction[0])
    
    # print("predictions: ", prediction[0])
    # print("predictions array: ", predictions_array)
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


_nsre = re.compile('([0-9]+)')
def natural_sort_key(s):
    return [int(text) if text.isdigit() else text.lower()
            for text in re.split(_nsre, s)]

def load_xml2image(xml_file, show_image=False):
    xml_root = xml.etree.ElementTree.parse(xml_file).getroot()
    data = list(xml_root[0][5].text.split('\n'))
    data = [x.strip('   ').split(' ') for x in data]

    large_list = []
    for small_list in data:
        large_list += small_list

    image = [int(x) for x in large_list[1:]]
    # image_max = np.max(image)
    image_max = 31800
    # print("image max: ", image_max)
    image = np.reshape(np.array(image), [240,320])
    norm_image = image*(1.0/image_max)
    # print(norm_image)

    if show_image:
        cv2.imshow('image',norm_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    
    return norm_image


def make_input_bundle(filenames_list):
    input_bundle = []
    for file_i in filenames_list:
        input_i = load_xml2image(file_i)
        input_bundle.append(input_i)
    
    # reverse the order of the elements in the list
    reversed(input_bundle)
    return np.array(input_bundle)


model_path = 'logdir_utkinect_2_simple_ff_6_96/model_epoch_80'
inference_model = InferenceModel(num_frames=cfg.num_frames, model_path=model_path)


# general solution
dataset_dir = '/media/tjosh/vault/UTKinectAction3D_depth_test'

directory_list = os.walk(dataset_dir)

actions = {"walk":0, "sitDown":1, "standUp":2, "pickUp":3, "carry":4, 
    "throw":5, "push":6, "pull":7, "waveHands":8,"clapHands":9}

label2actions = {0:"walk", 1:"sitDown", 2:"standUp", 3:"pickUp", 4:"carry", 
    5:"throw", 6:"push", 7:"pull", 8:"waveHands",9:"clapHands"}

with open('/media/tjosh/vault/UTKinectAction3D_depth_test/actionLabel.json') as f:
    json_file = json.load(f)

action_buffer_size = cfg.num_frames
action_counts = 0
predictions_array = np.ones(cfg.num_classes)
num_corrects = 0
for dirs in directory_list:
    file_dir = dirs[0]
    files_list = glob.glob(file_dir+'/*.xml')
    files_list.sort(key=natural_sort_key)
    previous_action = None
    
    action_buffer = []
    for filename in files_list:
        
        # subject_key = filename.split('\\')[-2]
        subject_key = filename.split('/')[-2]
        frame_no = int(re.split('depthImg|.xml',filename)[-2])
        subject_actions = json_file[subject_key]
        
        action_label = None
        for action in subject_actions:
            action_frame = subject_actions[action]
            # print("action: {}, action frame: {}.".format(actions[action], action_frame))
            if frame_no >= action_frame[0] and frame_no <= action_frame[1]:
                action_label = actions[action]
        else:
            if action_label == None:
                # print("No action label!")
                continue
        

        if previous_action != action_label:
            # this means a new action starts
            if previous_action==None:
              previous_action=action_label
            else:

              # report the evaluation for the last action and start a new evaluation for the next action
              final_action_prediction = np.argmax(predictions_array)
              print("\nthis action final: ", final_action_prediction)
              correct = int(final_action_prediction==previous_action)
              print("Correct: ", (final_action_prediction==previous_action))
              print("\n")
              num_corrects += correct
              # move on
              action_counts += 1
              previous_action = action_label
              action_buffer = []
              predictions_array = np.ones(cfg.num_classes)
        
        print("\n{}, frame {}.".format(subject_key, frame_no))
        print("action label: ", action_label)
        
        action_buffer.append(filename)

        if len(action_buffer) <= action_buffer_size:
          continue

        if len(action_buffer) > action_buffer_size:
            action_buffer.pop(0)

        # do stuff here:
        input_bundle = make_input_bundle(action_buffer)

        data_in = [np.transpose(input_bundle, (1, 2, 0))]

        # do inference with the present action buffer
        prediction = inference_model.sess.run(
                        inference_model.pred,
                        feed_dict={inference_model.inputs_pl: data_in, 
                                    inference_model.is_training_pl: False})
        
        predictions_array *= prediction[0]
        
        pred_action = np.argmax(prediction[0])
        print("action prediction: ", pred_action)
        # print("action counts: ", action_counts)

    # report the evaluation for the last action and start a new evaluation for the next action
    final_action_prediction = np.argmax(predictions_array)
    print("\nthis action final: ", final_action_prediction)
    correct = int(final_action_prediction == previous_action)
    print("Correct: ", (final_action_prediction == previous_action))
    print("\n")
    num_corrects += correct
    # move on
    action_counts += 1
    # previous_action = action_label
    action_buffer = []
    predictions_array = np.ones(cfg.num_classes)

        # print(np.shape(input_bundle))
print("Total action counts: ", action_counts)
print("Total correct predictions: ", num_corrects)
print("Final actions prediction accuracy: ", (num_corrects/float(action_counts)))
