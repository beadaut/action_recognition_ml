import glob
import cv2
import copy
import random

import numpy as np
import matplotlib.pyplot as plt

from mpl_toolkits.mplot3d import Axes3D
from scipy import ndimage
from copy import deepcopy
from sklearn.utils import shuffle

"""
Test the util functions
"""

def load_txt_filenames(filenames_txt_file, validation_split=0):
    txt_file = open(filenames_txt_file, "r")
    labels_present = False
    all_line_txt_list = []
    for line in txt_file:
      line_txt_list = line.rstrip('\n').split(' ')
      all_line_txt_list.append(line_txt_list)
    else:
      if len(line_txt_list) > 1:
        labels_present = True
    
    size_of_data = len(all_line_txt_list)
    print("size of data: ",size_of_data)
    if labels_present:
      all_filename_i = []
      all_label_i = []
      for i in range(size_of_data):
        filename_i = all_line_txt_list[i][0]
        label_i = all_line_txt_list[i][-1]
        
        # we want only the depth images: "K_"
        filename_i = filename_i.replace("M_", "K_")
        if "K_" in filename_i:
          all_filename_i.append(filename_i)
          all_label_i.append(label_i)
      
      return all_filename_i, all_label_i
    else:
      all_filename_i = []
      for i in range(size_of_data):
        filename_i = all_line_txt_list[i][0]

        # we want only the depth images: "K_"
        filename_i = filename_i.replace("M_", "K_")
        if "K_" in filename_i:
          all_filename_i.append(filename_i)
      return all_filename_i

    
    # all_files = None
    # # all_files = glob.glob(filenames_txt_file)
    # # all_files = shuffle(all_files)

    # # total_files_number = len(all_files)
    # # print ("Number of dataset samples: ",total_files_number)

    # if validation_split:
    #   val_split_point = int(validation_split*total_files_number)
    #   validation_files = all_files[:val_split_point]
    #   train_files = all_files[val_split_point:]
    #   print ("Number of training samples: ",len(train_files))
    #   print ("Number of validation samples: ",len(validation_files))
    #   return train_files, validation_files
    # else:
    #   return all_files

def read_clip_to_frames(clip, skip_frame=0, show_frames=False):
  """
  load a video clip and split the clip to frames
  params:
    clip: string; filename of the video clip
    skip_frame: int; interval of frames to skip while loading
  """
  cap = cv2.VideoCapture(clip)

  frames = []
  ret=True
  while(ret):
    # Capture frame-by-frame
    ret, frame = cap.read()
    
    # print("ret: ",ret)
    try:
      frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
      frames.append(frame)  
      
    except Exception as e:
      pass
  
  # Display the resulting frame
  if show_frames:
    for frame_i in frames:
      cv2.imshow('frame', frame_i)
      if cv2.waitKey(10) & 0xFF == ord('q'):
          break
    cv2.destroyAllWindows()

  # When everything done, release the capture
  cap.release()
  return frames


def read_clip_to_frame_bundles(clip, steps=5):
  clips_array = read_clip_to_frames(clip)

  input_bundle = []
  all_input_bundle = []
  for i, frame in enumerate(clips_array):
    if i < steps:
      continue
    for x in range(steps):
      input_i_x = clips_array[i-x]
    # print("shape of input_i_x", np.shape(input_bundle))
    input_bundle.append(input_i_x)
    if len(input_bundle)>steps:
      all_input_bundle.append(input_bundle)
      input_bundle.pop(0)
  
  # print("all input bundle shape: ", np.shape(all_input_bundle))
  # return np.array(all_input_bundle)
  return all_input_bundle

def transfrom_input(inputs, M):
  trans_inputs = []
  for i in range(np.shape(inputs)[0]):
    image = ndimage.affine_transform(inputs[i], M)
    trans_inputs.append(image)
  
  return np.array(trans_inputs)



class DataGenerator(object):
  """
  Class for creating a datset generator
  """

  def __init__(self, dataset_file_list, dataset_labels_list, batch_size=100, steps=6, augment=True):
    super(DataGenerator, self).__init__()

    self.augment = augment
    self.steps = steps
    self.all_dataset = dataset_file_list
    self.all_labels = dataset_labels_list
    self.all_dataset, self.all_labels = shuffle(self.all_dataset, self.all_labels)
    self.data_size = len(self.all_dataset)

    self.batch_size = batch_size
    
    self.iters_per_epoch = self.data_size//self.batch_size
    
    print("\nDataset loaded...")
    print("Size of dataset = ", self.data_size)

    # create generator
    self.generator = self.generate()

  def generate(self):

    while True:
      # initializations
      inputs_batch = []
      labels_batch = []

      for i in range(self.data_size):
        # print("\nsample: ", i)
        # current_file = self.all_dataset[i].replace("\\", "/")
        # print(self.all_dataset[i])
        current_file = "/media/tjosh/vault/chalearn_dataset/IsoGR_2016/train/" +str(self.all_dataset[i])
        # print("current file ", current_file)
        
        if len(labels_batch) > self.batch_size*5:

          # return generated here
          yield_inputs, yield_labels = shuffle(inputs_batch, labels_batch)
          yield_inputs = np.transpose(yield_inputs[:self.batch_size], (0,2,3,1))
          yield yield_inputs, yield_labels[:self.batch_size] # can turn this into numpy arrays later
          inputs_batch = []
          labels_batch = []

        current_bundle = read_clip_to_frame_bundles(current_file, steps=self.steps)

        label_i = int(self.all_labels[i])
        if label_i==249:
          label_i=0
        current_labels = [label_i]*len(current_bundle)
        
        # print("Filename: {}, Label: {}".format(self.all_dataset[i], current_labels))
        inputs_batch = inputs_batch+current_bundle
        labels_batch = labels_batch+current_labels
      
      # reshuffle the dataset after every epoch
      self.all_dataset, self.all_labels = shuffle(self.all_dataset, self.all_labels)


def trim_train_data():
  load_dir = "train"
  data_glob_dir = "/media/tjosh/vault/chalearn_dataset/IsoGR_2016/"
  filenames, labels = load_txt_filenames(
      data_glob_dir+load_dir+"/"+load_dir+"_list.txt")

  print("finished reading text file")
  print("now reading videos. This will take some time.")
  dataset_len = len(filenames)

  filenames_list = []
  labels_list = []
  for data_i in range(dataset_len):
    # print("data {} of {}".format(data_i, dataset_len))
    one_file = data_glob_dir+load_dir+"/" + \
        filenames[data_i]

    frames = read_clip_to_frames(one_file)

    frame_len = np.shape(frames)[0]
  
    if frame_len > 10:
      filenames_list.append(filenames[data_i])
      labels_list.append(labels[data_i])
  
  final_dataset_len = len(filenames_list)
  print("len of remaining data: ", final_dataset_len)
  
  # print(data_glob_dir+load_dir+"_filenames")
  # print(data_glob_dir+load_dir+"_labels")

  train_cut = int(final_dataset_len*0.2)
  filenames_list, labels_list = shuffle(filenames_list, labels_list)

  np.save(data_glob_dir+load_dir+"_filenames_2", filenames_list[train_cut:])
  np.save(data_glob_dir+load_dir+"_labels_2", labels_list[train_cut:])

  np.save(data_glob_dir+load_dir+"_validation_filenames_2", filenames_list[:train_cut])
  np.save(data_glob_dir+load_dir+"_validation_labels_2", labels_list[:train_cut])

    
    


def test_loader():
  # filenames, labels = load_txt_filenames(
  #     "/media/tjosh/vault/chalearn_dataset/IsoGR_2016/train/train_list.txt")
  
  filenames, labels = load_txt_filenames(
      "/media/tjosh/vault/chalearn_dataset/IsoGR_2016/test/test_list.txt")

  # print(type(filenames))
  frame_len_list = []
  dataset_len = len(filenames)
  for data_i in range(dataset_len):
    # print("data {} of {}".format(data_i, dataset_len))
    one_file = "/media/tjosh/vault/chalearn_dataset/IsoGR_2016/test/" + \
        filenames[data_i]

    # print("filename: ",one_file)
    # print("label: ",labels[data_i])
    # print("\n")

    # frames = read_clip_to_frame_bundles(one_file, steps=10)
    frames = read_clip_to_frames(one_file)

    frame_len = np.shape(frames)[0]
    # print(frame_len)
    frame_len_list.append(frame_len)
    # break

  print("list_shape: ", np.shape(frame_len_list))
  print("list minimum: ", np.min(frame_len_list))
  print("list maximum: ", np.max(frame_len_list))
  print("list average: ", np.average(frame_len_list))

def test_generator():
  # filenames, labels = load_txt_filenames(
  #     "/media/tjosh/vault/chalearn_dataset/IsoGR_2016/train/train_list.txt")

  filenames = np.load(
      '/media/tjosh/vault/chalearn_dataset/IsoGR_2016/train_filenames.npy')
  labels = np.load(
      '/media/tjosh/vault/chalearn_dataset/IsoGR_2016/train_labels.npy')

  data_gen = DataGenerator(filenames, labels, batch_size=64)
  new_data_batch = next(data_gen.generator)
  x = new_data_batch[0]
  y = new_data_batch[1]
  # print("size of dataset batch: ", np.shape(x))
  # print(y)
  # for i in range(20):
  #   print(x[i])
  #   # show_sample(x[i])
  #   print("shape of x[i]: ", np.shape(x[i]))
  #   print(y[i])


def test_logic():
  import matplotlib.pyplot as plt
  import numpy as np
  from matplotlib import colors
  from matplotlib.ticker import PercentFormatter

  # filenames, labels = load_txt_filenames(
  #     "/media/tjosh/vault/chalearn_dataset/IsoGR_2016/train/train_list.txt")

  filenames = np.load(
      '/media/tjosh/vault/chalearn_dataset/IsoGR_2016/train_validation_filenames.npy')
  labels = np.load(
      '/media/tjosh/vault/chalearn_dataset/IsoGR_2016/train_validation_labels.npy')
    
  classes = 249
  labels_hist = [0]*classes
  for i in range(len(labels)):
    # print("Filename: {}, Label: {}".format(filenames[i], labels[i]))
    # print(type(labels[i]))
    label_i = int(labels[i])-1
    if label_i > classes:
      print("outlier: ", label_i)
    labels_hist[label_i]+=1

  print("average count: ", np.average(labels_hist))
  print("max count: ", np.max(labels_hist))
  print("min count: ", np.min(labels_hist))

  fig, ax = plt.subplots()
  ax.bar(list(range(classes)), labels_hist)

  # n, bins, patches = plt.hist(x=labels_hist, bins=n_bins)
  # n, bins, patches = plt.hist(x=labels_hist, bins="auto")
  plt.show()

  

if __name__ == '__main__':
  # test_generator()
  # test_logic()
  # test_loader()
  trim_train_data()
