import sys
import numpy as np

train_subset = sys.argv[1]
crop_window_len = np.int(sys.argv[2])

crop_linear_len = (crop_window_len*2 + 1) * (crop_window_len*2 + 1) * (crop_window_len*2 + 1)

saving_mm_name = str(crop_window_len * 2 +1) + 'mm'

import random

import os


root_path = '/LUNA16/Train_data/' + saving_mm_name + '/'+train_subset + '/'
saving_path = root_path + 'much_data/'
if not os.path.exists(saving_path):
  os.makedirs(saving_path)

nodule_data = np.load(saving_path + 'nodule_data.npy')
nodule_label = np.load(saving_path + 'nodule_label.npy')
len_nodule = len(nodule_data)

augment_nodule_data = np.load(saving_path + 'augment_nodule_data.npy')
augment_nodule_label = np.load(saving_path + 'augment_nodule_label.npy')
aug_idx = np.random.choice(len(augment_nodule_data), 100 * len_nodule, replace = False)

non_nodule_boxes_data = np.load(saving_path + 'non_nodule_boxes_data.npy')
non_nodule_boxes_label = np.load(saving_path + 'non_nodule_boxes_label.npy')
boxes_idx = np.random.choice(len(non_nodule_boxes_data), 100 * len_nodule, replace = False)


much_data = np.append(nodule_data, augment_nodule_data[aug_idx], axis = 0)
much_label = np.append(nodule_label, augment_nodule_label[aug_idx], axis = 0)

much_data = np.append(much_data, non_nodule_boxes_data[boxes_idx], axis = 0)
much_label = np.append(much_label, non_nodule_boxes_label[boxes_idx], axis = 0)


idx = np.array(range(len(much_data)))
random.shuffle(idx)
random.shuffle(idx)
much_data_shuffled = much_data[idx]
much_label_shuffled = much_label[idx]

np.save(saving_path + 'subsampled_much_data.npy', much_data_shuffled)
np.save(saving_path + 'subsampled_much_label.npy', much_label_shuffled)
 
