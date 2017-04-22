import sys
import numpy as np

train_subset = sys.argv[1]
crop_window_len = np.int(sys.argv[2])

crop_linear_len = (crop_window_len*2 + 1) * (crop_window_len*2 + 1) * (crop_window_len*2 + 1)

saving_mm_name = str(crop_window_len * 2 +1) + 'mm'

root_path = '/LUNA16/Train_data/' + saving_mm_name + '/'+train_subset + '/'
saving_path = root_path + 'much_data/'

subsampled_much_data = np.load(saving_path + 'subsampled_much_data.npy')
# subsampled_much_data_mean = np.int(np.rint(np.mean(subsampled_much_data)))
norm_subsampled_much_data = subsampled_much_data - np.int(-635)

np.save(saving_path + 'norm_subsampled_much_data.npy', norm_subsampled_much_data)
# np.save(saving_path + 'subsampled_much_data_mean.npy', subsampled_much_data_mean)
 
