import glob
import numpy as np

from sklearn.utils import shuffle


# cut make training set and validation set
all_data = glob.glob('/media/tjosh/vault/MSRAction3D/pc_npy_5/*.npy')
print(len(all_data))
size_of_data = len(all_data)
all_data = shuffle(all_data)
training_cut = all_data[int(0.33*size_of_data):]
validation_cut = all_data[:int(0.33*size_of_data)]
np.save('/media/tjosh/vault/MSRAction3D/pc_npy_5_training', training_cut)
np.save('/media/tjosh/vault/MSRAction3D/pc_npy_5_validation', validation_cut)
print('all saved!')
