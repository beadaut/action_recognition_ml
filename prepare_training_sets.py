import glob
import csv
import numpy as np
# import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

from sklearn.utils import shuffle


# cut make training set and validation set
all_data = glob.glob('/media/tjosh/vault/MSRAction3D/npy_5/*.npy')
all_data = shuffle(all_data)


set_1_labels = [2,3,5,6,10,13,18,20]
set_2_labels = [1,4,7,8,9,11,12,14]
set_3_labels = [6,14,15,16,17,18,19,20]

# for set 1
set_1_filenames = []
for data_sample in all_data:
  label_i = int(data_sample.split('/')[-1].split('_')[0].split('a')[1])
  if label_i in set_1_labels:
    set_1_filenames.append(data_sample)

np.save('/media/tjosh/vault/MSRAction3D/npy_5_set_1', set_1_filenames)
print("size of set 1: ", len(set_1_filenames))



# for set 2
set_2_filenames = []
for data_sample in all_data:
  label_i = int(data_sample.split('/')[-1].split('_')[0].split('a')[1])
  if label_i in set_2_labels:
    set_2_filenames.append(data_sample)

np.save('/media/tjosh/vault/MSRAction3D/npy_5_set_2', set_2_filenames)
print("size of set 2: ", len(set_2_filenames))

# for set 3
set_3_filenames = []
for data_sample in all_data:
  label_i = int(data_sample.split('/')[-1].split('_')[0].split('a')[1])
  if label_i in set_3_labels:
    set_3_filenames.append(data_sample)


np.save('/media/tjosh/vault/MSRAction3D/npy_5_set_3', set_3_filenames)
print("size of set 3: ", len(set_3_filenames))


# print(len(all_data))
# size_of_data = len(all_data)
# all_data = shuffle(all_data)
# training_cut = all_data[int(0.2*size_of_data):]
# validation_cut = all_data[:int(0.2*size_of_data)]
# np.save('/media/tjosh/vault/MSRAction3D/npy_5_training', training_cut)
# np.save('/media/tjosh/vault/MSRAction3D/npy_5_validation', validation_cut)
# print('all saved!')