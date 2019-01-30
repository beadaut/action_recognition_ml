import glob
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

def rotate_point_cloud(batch_data):

    """ Randomly rotate the point clouds to augument the dataset
        rotation is per shape based along up direction
        Input:
          BxNx3 array, original batch of point clouds
        Return:
          BxNx3 array, rotated batch of point clouds
    """
    rotated_data = np.zeros(batch_data.shape, dtype=np.float32)
    for k in range(batch_data.shape[0]):
        rotation_angle = np.random.uniform() * 2 * np.pi
        cosval = np.cos(rotation_angle)
        sinval = np.sin(rotation_angle)
        rotation_matrix = np.array([[cosval, 0, sinval],
                                    [0, 1, 0],
                                    [-sinval, 0, cosval]])
        shape_pc = batch_data[k, ...]
        rotated_data[k, ...] = np.dot(shape_pc.reshape((-1, 3)), rotation_matrix)
    return rotated_data

def rotate_point_cloud_by_angle(batch_data, rotation_angle):
    """ Rotate the point cloud along up direction with certain angle.
        Input:
          BxNx3 array, original batch of point clouds
        Return:
          BxNx3 array, rotated batch of point clouds
    """
    rotated_data = np.zeros(batch_data.shape, dtype=np.float32)
    for k in range(batch_data.shape[0]):
        #rotation_angle = np.random.uniform() * 2 * np.pi
        cosval = np.cos(rotation_angle)
        sinval = np.sin(rotation_angle)
        rotation_matrix = np.array([[cosval, 0, sinval],
                                    [0, 1, 0],
                                    [-sinval, 0, cosval]])
        shape_pc = batch_data[k, ...]
        rotated_data[k, ...] = np.dot(shape_pc.reshape((-1, 3)), rotation_matrix)
    return rotated_data

def jitter_point_cloud(batch_data, sigma=0.01):
    """ Randomly jitter points. jittering is per point.
        Input:
          BxNx3 array, original batch of point clouds
        Return:
          BxNx3 array, jittered batch of point clouds
    """
    clip = sigma
    # print("\nshape: ", np.shape(batch_data))
    # B, N, C = np.shape(batch_data)

    jittered_data = np.zeros_like(batch_data)
    jittered_data[:,:,0,:]+= (sigma * np.random.rand()) # x axis
    jittered_data[:,:,1,:]+= (sigma * np.random.rand()) # y axis
    jittered_data[:,:,2,:]+= (sigma * np.random.rand()) # z axis
    # assert(clip > 0)
    # jittered_data = np.clip(sigma * np.random.randn(B, N, C), -1*clip, clip)
    # print("jittered_data: ", jittered_data)
    # print("non_zero: ", np.nonzero(batch_data))
    jittered_data += batch_data
    return jittered_data

    
def load_npy_filenames(glob_dir, validation_split=0):
    all_files = glob.glob(glob_dir)
    all_files = shuffle(all_files)

    total_files_number = len(all_files)
    print ("Number of dataset samples: ",total_files_number)

    if validation_split:
      val_split_point = int(validation_split*total_files_number)
      validation_files = all_files[:val_split_point]
      train_files = all_files[val_split_point:]
      print ("Number of training samples: ",len(train_files))
      print ("Number of validation samples: ",len(validation_files))
      return train_files, validation_files
    else:
      return all_files


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

  def __init__(self, dataset_file_list, batch_size=100, reshape_input=None, augment=True):
    super(DataGenerator, self).__init__()

    self.augment = augment
    self.all_dataset = dataset_file_list
    self.all_dataset = shuffle(self.all_dataset)
    self.data_size = len(self.all_dataset)

    self.batch_size = batch_size
    
    self.iters_per_epoch = self.data_size//self.batch_size
    
    print("\nDataset loaded...")
    print("Size of dataset = ", self.data_size)

    # create generator
    self.generator = self.generate()

    if reshape_input:
      self.input_shape = copy.deepcopy(reshape_input)
      self.features = self.input_shape[0] # number of feature to use
      self.input_shape[0] = -1
    else: self.input_shape = None

  def generate(self):

    # batches_per_epoch = self.data_size/self.data_size
    # batch_no = 0
    while True:
      # initializations
      inputs_batch = []
      labels_batch = []

      # transformation
      transf_x = int(random.random()*100) - 50
      M = np.float32([[1, 0, 0], [0, 1, transf_x]])

      for i in range(self.data_size):
        # print("\nsample: ", i)
        # current_file = self.all_dataset[i].replace("\\", "/")
        # print(self.all_dataset[i])
        current_file = str(self.all_dataset[i])
        # print(current_file)
        if len(labels_batch) > self.batch_size-1:

          # return generated here
          
          yield inputs_batch, labels_batch
          inputs_batch = []
          labels_batch = []

          # transformation
          transf_x = int(random.random()*50) - 25
          M = np.float32([[1, 0, 0], [0, 1, transf_x]])
          
        
        # print("file: ", current_file)
        input_i = np.load(current_file) # for npy files

        # if self.augment:
        #   input_i = transfrom_input(input_i, M)

        # input_i = np.flipud(input_i) # this is a fix on the order of array, need to use a more effecient solution
        # print("shape of input_i", np.shape(input_i))
        input_i = np.transpose(input_i, (1,2,0)) # transpose the input to the appropriate shape
        
        # label_i = int(current_file.split("/")[-2].split("_")[-1])
        # print("current file: ", current_file)

        # label_i = int(current_file.split('/')[-1].split('_')[0].split('a')[1]) # for msraction 3d
        # if label_i==20:
        #   label_i=0
        
        label_i = int(current_file.split('\\')[-1].split('.')[0].split('_')[-1]) # for utkinect action 3d
        # print('label: ', label_i)
        
        inputs_batch.append(input_i)
        labels_batch.append(label_i)
      
      # reshuffle the dataset after every epoch
      self.all_dataset = shuffle(self.all_dataset)


class TripletGenerator(object):
  """
  Class for creating a triplet datset generator
  """

  def __init__(self, dataset_file_list, classes, batch_size=100,
               reshape_input=None, seek_anchor=None, seek_neg=None):
    super(TripletGenerator, self).__init__()

    self.seek_anchor = seek_anchor
    self.seek_neg = seek_neg
    self.set_size = 3

    self.all_dataset = shuffle(dataset_file_list)
    self.dataset_dict = dataset_2_dict(dataset_file_list, classes)
    self.data_size = len(self.all_dataset)

    self.batch_size = batch_size

    self.iters_per_epoch = self.data_size//self.batch_size

    # self.classes_list = [x for x in range(classes)]
    self.classes_list = classes
    # self.classes_list = [6,7,8]

    print("\nDataset loaded...")
    print("Size of dataset = ", self.data_size)

    # create generator
    self.generator = self.generate()

    if reshape_input:
      self.input_shape = copy.deepcopy(reshape_input)
      self.features = self.input_shape[0]  # number of feature to use
      self.input_shape[0] = -1
    else:
      self.input_shape = None

  def generate(self):

    # initializations
    now_key = 0
    triplet_inputs = []
    triplet_labels = []
    triplet = pick_triplet(
        self.classes_list, seek_anchor=self.seek_anchor, seek_neg=self.seek_neg)
    inputs_batch = []
    labels_batch = []
    while True:

      for i in range(self.data_size):
        # print("\nsample: ", i)
        try:
          # get data sample
          # print("filename: ", self.all_dataset[i][0])
          filename = self.all_dataset[i].replace('\\', '/')
          # filename = self.all_dataset[i].replace('/','\\')
          input_i = np.load(filename)
          # print("filename: ", filename)
          
          # this is a fix on the order of array, need to use a more effecient solution
          # input_i = np.flipud(input_i) # for others

          
          # transpose the input to the appropriate shape
          # input_i = np.transpose(input_i, (1, 2, 0))

          # label_i = int(filename.split("/")[-2].split("_")[-1]) # for another

          input_i = np.transpose(input_i, (1,2,0))
          label_i = int(filename.split('/')[-1].split('_')[0].split('a')[1])
          # label_i = int(re.split('cloud_|_label_|_reached_|.ply',
          #                        self.all_dataset[i])[-1].split("-")[0])

          # print("shape of input_i: ", np.shape(input_i))
          # print("label of input_i: ", label_i)
        except Exception as e:
          print(e)
          continue

        if label_i == triplet[now_key]:
          triplet_inputs.append(input_i)
          triplet_labels.append(label_i)
          print("label_i: ",label_i)
          # print("now key: ",now_key)
          # triplet_inputs.append(label_i) # to test which input class is sent
          now_key += 1

        if now_key == self.set_size:
          # # insert datset here
          # print("labels_batch: ",triplet_labels)
          inputs_batch.append(triplet_inputs)
          labels_batch.append(triplet_labels)
          triplet_inputs = []
          triplet_labels = []
          now_key = 0

          # including this here allows it to reset the support set everytime it is filled up
          triplet = pick_triplet(
              self.classes_list, seek_anchor=self.seek_anchor, seek_neg=self.seek_neg)

        if len(labels_batch) > self.batch_size-1:

          # return generated here
          # print("shape of input batch: ", np.shape(inputs_batch))
          # inputs_batch = np.transpose(np.array(inputs_batch), (0, 1, 3, 4, 2))
          yield inputs_batch, labels_batch
          inputs_batch = []
          labels_batch = []
          triplet_inputs = []
          now_key = 0

      # reshuffle the dataset after every epoch
      self.all_dataset = shuffle(self.all_dataset)
      if now_key >= self.set_size+1:
        inputs_batch = []
        labels_batch = []
        triplet = pick_triplet(
            self.classes_list, seek_anchor=self.seek_anchor, seek_neg=self.seek_neg)

class NewTripletGenerator(object):
  """
  Class for creating a triplet datset generator
  """

  def __init__(self, dataset_file_list, classes, batch_size=100,
               reshape_input=None, seek_anchor=None, seek_neg=None):
    super(NewTripletGenerator, self).__init__()

    self.seek_anchor = seek_anchor
    self.seek_neg = seek_neg
    self.set_size = 3

    self.all_dataset = shuffle(dataset_file_list)
    self.dataset_dict = dataset_2_dict(dataset_file_list, classes)
    self.data_size = len(self.all_dataset)

    self.batch_size = batch_size

    self.iters_per_epoch = self.data_size//self.batch_size

    # self.classes_list = [x for x in range(classes)]
    self.classes_list = classes
    # self.classes_list = [6,7,8]

    print("\nDataset loaded...")
    print("Size of dataset = ", self.data_size)

    # create generator
    self.generator = self.generate()

    if reshape_input:
      self.input_shape = copy.deepcopy(reshape_input)
      self.features = self.input_shape[0]  # number of feature to use
      self.input_shape[0] = -1
    else:
      self.input_shape = None

  def generate(self):

    # initializations
    triplet_inputs = []
    triplet_labels = []
    inputs_batch = []
    labels_batch = []
    while True:
      triplet = pick_triplet(
          self.classes_list, seek_anchor=self.seek_anchor, seek_neg=self.seek_neg)
      triplet_inputs = []
      triplet_labels = []
      for i, t_label in enumerate(triplet):
        # print("t label: ", t_label)
        filename = random.choice(self.dataset_dict[t_label])
        input_i = np.load(filename)
        input_i = np.transpose(input_i, (1,2,0))

        label_i = int(filename.split('/')[-1].split('_')[0].split('a')[1])
        # label_i = int(filename.split('\\')[-1].split('_')[0].split('a')[1])
        
        triplet_inputs.append(input_i)
        triplet_labels.append(label_i)
      
      inputs_batch.append(triplet_inputs)
      labels_batch.append(triplet_labels)
      triplet_inputs = []
      triplet_labels = []


      if len(labels_batch) > self.batch_size-1:

        # return generated here
        # print("shape of input batch: ", np.shape(inputs_batch))
        # inputs_batch = np.transpose(np.array(inputs_batch), (0, 1, 3, 4, 2))
        yield inputs_batch, labels_batch
        inputs_batch = []
        labels_batch = []
        triplet_inputs = []

      # reshuffle the dataset after every epoch
      self.all_dataset = shuffle(self.all_dataset)

            
def pick_triplet(labels_in, seek_anchor=None, seek_neg=None):
  """
  Randomly picks a triplet consisting of an anchor, negative and positive label
  """
  assert len(labels_in) >= 3, "Value of set size must be less than or equal to 3!"
  labels = deepcopy(labels_in)

  if seek_anchor:
    anchor = seek_anchor
  else:
    anchor = random.choice(labels)
    del(labels[labels.index(anchor)])  # delete the already used element

  if seek_neg == None:
    negative = random.choice(labels)
  else:
    negative = seek_neg

  if anchor==negative:
    return pick_triplet(labels_in, seek_anchor, seek_neg) # in case there is a repetition

  return [anchor, negative, anchor]

def dataset_2_dict(files_list, labels_list):
  # print(len(files_list))
  dataset_dict = {}
  for label_i in labels_list:
    dataset_dict[label_i] = []
  
  # print("labels dict: ", dataset_dict)
  for file_i in files_list:
    # label_i = int(file_i.split('\\')[-1].split('_')[0].split('a')[1])
    label_i = int(file_i.split('/')[-1].split('_')[0].split('a')[1])
    # print("label: ", label_i)
    if label_i in labels_list:
      dataset_dict[label_i].append(file_i)
  
  # print(random.choice(dataset_dict[2]))
  # print("scaled throught.................")
  return dataset_dict



def test_triplet_generator():
  """ Test the dataset generator module"""
  # dataset_glob_path = 'd:/datasets/MSRAction3D/npy_5/*.npy'
  dataset_glob_path = '/media/tjosh/vault/MSRAction3D/npy_5/*.npy'
  train_dataset, validation_dataset = load_npy_filenames(
      dataset_glob_path, validation_split=0.2)
  batch_size = 10
  classes_list = [2,4,5,6,7,9,10,11,12,13,14,16,17,19,20]
  test_data = NewTripletGenerator(
      train_dataset, classes=classes_list, batch_size=batch_size)

  for i in range(10):
    print("\nEpoch: ", i+1)
    new_data_batch = next(test_data.generator)
    x = new_data_batch[0]
    y = new_data_batch[1]
    print("shape of batch dataset: ", np.shape(x))
    for i in range(batch_size):
      # print(np.shape(x[i]))
      # print(x[i])
      print(y[i])


def show_sample(x):
    # create plot object
    print("shape of x: ", np.shape(x))
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.view_init(0, 270)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])

    # fill the plot with data (auto)
    for i, (c, m) in enumerate([('r', 'o'), ('b', '^'), ('y', 'X'), ('g', 'v')]):
        xs = x[:, :, i][:, 0]
        ys = x[:, :, i][:, 1]
        zs = x[:, :, i][:, 2]
        # print('max of zs: ', np.max(zs))
        # print('min of zs: ', np.min(zs))
        ax.scatter(xs, ys, zs, s=1.5, c=c, marker=m)
    
    plt.show()

def test_data_gen():
  # train_dataset = np.load('d:/datasets/MSRAction3D/training.npy')
  train_dataset = glob.glob('d:/datasets/MSRAction3D/npy_5/*.npy')
  data_gen = DataGenerator(train_dataset, batch_size=20)
  new_data_batch = next(data_gen.generator)
  x = new_data_batch[0]
  y = new_data_batch[1]
  print("size of dataset batch: ", np.shape(x))
  # for i in range(20):
  #   print(x[i])
  #   # show_sample(x[i])
  #   print("shape of x[i]: ", np.shape(x[i]))
  #   print(y[i])

def test_pick_triplet():

  classes_list = [0,2,4,5,6,7,9,10,11,12,13,14,16,17,19]
  triplets = pick_triplet(classes_list, seek_anchor=None, seek_neg=None)
  print(triplets)


if __name__ == '__main__':
  # train_dataset, validation_dataset = load_npy_filenames("dataset/hand_controls_npy_4_1024/**/*.npy", validation_split=0.2)
  # data_gen = DataGenerator(train_dataset, batch_size=20)
  # new_data_batch = next(data_gen.generator)
  # x = new_data_batch[0]
  # y = new_data_batch[1]
  # print("size of dataset batch: ", np.shape(x))
  # for i in range(20):
  #   # print(x[i])
  #   print("shape of x[i]: ", np.shape(x[i]))
  #   # print(y[i])

  # test_data_gen()
  test_triplet_generator()
  # for i in range(20):
  #   test_pick_triplet()

  # dataset_glob_path = glob.glob('d:/datasets/MSRAction3D/npy_5/*.npy')
  # classes_list = [2,4,5,6,7,9,10,11,12,13,14,16,17,19,20]
  # dataset_2_dict(dataset_glob_path, classes_list)
