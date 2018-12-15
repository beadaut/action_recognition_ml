import os
import glob
import copy
import numpy as np

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


def load_pcd_py3(filename, truncate=True, max_points=1024):
  cloud = []
  with open(filename) as file:
    for i, line in enumerate(file):
      if i < 11:
        continue
      else:
        line_split = (line.split(" ")[:3])
        cloud.append([float(line_split[0]), float(line_split[1]), float(line_split[2])])
  
  if truncate:
      pcd_array = np.zeros((max_points, 3))
      # print("in max points")
      temp_array = np.array(cloud)
      np.random.shuffle(temp_array)
      # pcd_array = pcd_array[:max_points]
      if np.shape(temp_array)[0]<max_points:
          pcd_array[:np.shape(temp_array)[0]]=temp_array
      else:
          pcd_array = temp_array[:max_points]
  else:
      # cloud_len = cloud.size
      # pcd_array = np.zeros((np.shape(cloud)[0], 3))
      pcd_array=np.array(cloud)
  
  return pcd_array


class MultiArrayGenerator(object):
  """
  Generate pcd multiarrays from a pcd files directory
  """
  def __init__(self, glob_dir, max_points=1024, time_steps=4, truncate=True):
    super(MultiArrayGenerator, self).__init__()
    self.all_files = glob.glob(glob_dir)
    self.all_files.sort(key=os.path.getmtime)
    self.time_steps = time_steps
    self.max_points = max_points
    self.truncate = truncate
    self.number_of_files = len(self.all_files)
    print("Number of files: ", self.number_of_files)

    self.generator = self.generate()

  def generate(self):
    sample_array = []
    while True:
      for i, _ in enumerate(self.all_files):
          for k in range(self.time_steps):
              sample_array.append(load_pcd_py3(self.all_files[i+k], max_points=self.max_points))
              state_now = self.all_files[i+k].split("_")[-1].split(".")[0]
              
              if state_now == "end" and len(sample_array)<self.time_steps:
                  sample_array = []
                  break
          if len(sample_array) < self.time_steps:
            continue
          else:
            yield sample_array


class DataGenerator(object):
  """
  Class for creating a datset generator
  """

  def __init__(self, dataset_file_list, batch_size=100, reshape_input=None):
    super(DataGenerator, self).__init__()

    self.all_dataset = dataset_file_list
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
      
      for i in range(self.data_size):
        # print("\nsample: ", i)
        current_file = self.all_dataset[i].replace("\\", "/")
        if len(labels_batch) > self.batch_size-1:

          # return generated here
          yield inputs_batch, labels_batch
          inputs_batch = []
          labels_batch = []
        
        # print("file: ", current_file)
        # input_i = np.load(current_file) # for npy files
        input_i = load_files_generator(current_file) # for generator from pcd files

        input_i = np.flipud(input_i) # this is a fix on the order of array, need to use a more effecient solution
        input_i = np.transpose(input_i, (1,2,0)) # transpose the input to the appropriate shape
        
        label_i = int(current_file.split("/")[-2])
        
        inputs_batch.append(input_i)
        labels_batch.append(label_i)
      
      # reshuffle the dataset after every epoch
      self.all_dataset = shuffle(self.all_dataset)
  

if __name__ == '__main__':
  # train_dataset, validation_dataset = load_npy_filenames("dataset/hand_controls_raw_npy_4_1024/**/*.npy", validation_split=0.2)
  # data_gen = DataGenerator(train_dataset, batch_size=20)
  # new_data_batch = next(data_gen.generator)
  # x = new_data_batch[0]
  # y = new_data_batch[1]
  # print("size of dataset batch: ", np.shape(x))
  # for i in range(20):
  #   # print(x[i])
  #   # print("shape of x[i]: ", np.shape(x[i]))
  #   print(y[i])

  train_data_files = "dataset/hand_controls/**/*.pcd"
  train_data_gen = MultiArrayGenerator(train_data_files)
  new_multi_array = next(train_data_gen.generator)
  print("shape of multiarray: ", np.shape(new_multi_array))