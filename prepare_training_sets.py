import glob
import csv
import numpy as np
# import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

from sklearn.utils import shuffle


# cut make training set and validation set
all_data = glob.glob(
    # 'd:/datasets/MSRAction3D/npy_5/*.npy')
    # 'dataset/new_npy_5_1024/*.npy')
    '/media/tjosh/vault/MSRAction3D/npy_5/*.npy')
all_data = shuffle(all_data)

# # for all data together:
# size_of_data = len(all_data)
# cut_fraction = 0.67
# training_cut = all_data[ : int(cut_fraction*size_of_data)]
# validation_cut = all_data[int(cut_fraction*size_of_data) : ]
# np.save('/media/tjosh/vault/MSRAction3D/all_npy_5_t2_training', training_cut)
# np.save('/media/tjosh/vault/MSRAction3D/all_npy_5_t2_validation', validation_cut)
# print("Training set size: ", len(training_cut))


# # # set_1_labels = [2,3,5,6,10,13,18,20]
set_1_labels = [2,4,5,6,7,9,10,11,12,13,14,16,17,19,20]

# for set 1
set_1_filenames = []
for data_sample in all_data:
  label_i = int(data_sample.split('/')[-1].split('_')[0].split('a')[1])
  # label_i = int(data_sample.split('\\')[-1].split('_')[0].split('a')[1])
  if label_i in set_1_labels:
    set_1_filenames.append(data_sample)

data_size = len(set_1_filenames)
cut_point = int(0.80*data_size)
# np.save('/media/tjosh/vault/MSRAction3D/npy_5_set_1', set_1_filenames)
np.save('/media/tjosh/vault/MSRAction3D/one_shot_train', set_1_filenames[:cut_point])
np.save('/media/tjosh/vault/MSRAction3D/one_shot_test_for_known', set_1_filenames[cut_point:])
print("size of set 1: ", data_size)
print("size of training: ", cut_point)

# set_2_labels = [1, 4, 7, 8, 9, 11, 12, 14]

# # for set 2
# set_2_filenames = []
# for data_sample in all_data:
#   label_i = int(data_sample.split('/')[-1].split('_')[0].split('a')[1])
#   if label_i in set_2_labels:
#     set_2_filenames.append(data_sample)

# np.save('/media/tjosh/vault/MSRAction3D/npy_5_set_2', set_2_filenames)
# print("size of set 2: ", len(set_2_filenames))

# set_3_labels = [6, 14, 15, 16, 17, 18, 19, 20]

# # set_3_subjects = ['s01', 's02', 's03', 's04', 's05', 's06', 's07', 's08', 's09', 's10']
# set_3_subjects = ['s01', 's02', 's03', 's04', 's05']

# # for set 3
# set_3_filenames = []
# for data_sample in all_data:
#   label_i = int(data_sample.split('/')[-1].split('_')[0].split('a')[1])
#   subject_i = data_sample.split('/')[-1].split('_')[1]
#   if (subject_i in set_3_subjects) and (label_i in set_3_labels):
#     set_3_filenames.append(data_sample)


# np.save('/media/tjosh/vault/MSRAction3D/npy_5_set_3_train', set_3_filenames)
# print("size of set 3: ", len(set_3_filenames))
# # 20,000


# test_3_labels = [1, 4, 7, 8, 9, 11, 12, 14]

# test_3_subjects = ['s01', 's02', 's03', 's04', 's05', 's06', 's07', 's08', 's09', 's10'] # all subjects
# test_3_subjects = ['s02', 's04', 's06', 's08', 's10'] # validation
# test_3_subjects = ['s01', 's03', 's05', 's07', 's09'] # training

# # for set 3
# test_3_filenames = []
# for data_sample in all_data:
#   # label_i = int(data_sample.split('/')[-1].split('_')[0].split('a')[1])
#   subject_i = data_sample.split('/')[-1].split('_')[1]
#   if (subject_i in test_3_subjects):
#     test_3_filenames.append(data_sample)

# # np.save('/media/tjosh/vault/MSRAction3D/new_pc_npy_5_t3_training', test_3_filenames)
# np.save('dataset/new_pc_npy_5_t3_training', test_3_filenames)
# print("size of test: ", len(test_3_filenames))

# 20,000

# # for utkinect action 3d dataset ##################################################
# all_data = glob.glob(
#     '/media/tjosh/vault/UTKinectAction3D_depth_train/**/*.npy')
# all_data = shuffle(all_data)

# print(len(all_data))
# size_of_data = len(all_data)
# all_data = shuffle(all_data)
# training_cut = all_data[int(0.3*size_of_data):]
# validation_cut = all_data[:int(0.3*size_of_data)]
# np.save('d:/datasets/UTKinectAction3D_npy_5/training', training_cut)
# np.save('d:/datasets/UTKinectAction3D_npy_5/validation', validation_cut)
# print('all saved!')
