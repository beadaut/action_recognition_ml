import glob
import csv
# import cv2
import random
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

from scipy import ndimage
from sklearn.utils import shuffle

from convert_depth2pc import generate_pointcloud

labels_count = [0]*20

def csv2input_data(filename, time_steps=5, display_images=False, save_dir='npy'):
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
    print(len(all_data))

  all_data = np.reshape(all_data, [240*320,-1])
  # all_data = np.reshape(all_data, [-1,240,320])

  print(np.shape(all_data))
  print(np.shape(all_data[:,0]))
  # print(all_data[0])

  # time_steps = 5

  # plt.imshow(all_data[0])
  for i in range(np.shape(all_data)[-1]):
    if i < time_steps:
      continue
    input_filename = filename.split('.')[0].split('/')[1]
    input_filename = save_dir+'/'+input_filename+(str(i))
    # print('preprocessed: ', input_filename)
    input_bundle = []
    for x in range(time_steps):
      input_i_x = np.reshape(all_data[:,i-x], [240,320])
      input_bundle.append(input_i_x)
    # input_bundle = [np.reshape(all_data[:,i-x], [240,320]) for x in range(time_steps)]
    
    # saving a numpy file
    # np.save(input_filename, np.array(input_bundle))

    # saving as pointcloud file
    print("shape of raw data: ", np.shape(all_data[:,i]))
    input_data_i = np.reshape(all_data[:,i], [240,320])
    generate_pointcloud(input_data_i, 'filename_here')


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

  print('Shape of input bundle: ', np.shape(input_bundle))
  label = filename.split('/')[1].split('_')[0].split('a')[1]
  labels_count[int(label)-1]+=1
  # print('Lable: ', int(label))
  print('Lable count: ', labels_count)

# test:
filename = 'csv/a01_s01_e03_sdepth.csv'
csv2input_data(filename, time_steps=4, display_images=False)

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