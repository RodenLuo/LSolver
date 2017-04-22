import sys
import numpy as np

train_subset = sys.argv[1]
crop_window_len = np.int(sys.argv[2])

crop_linear_len = (crop_window_len*2 + 1) * (crop_window_len*2 + 1) * (crop_window_len*2 + 1)

saving_mm_name = str(crop_window_len * 2 +1) + 'mm'

import random

from glob import glob
import os


root_path = '/LUNA16/Train_data/' + saving_mm_name + '/'+train_subset + '/'
nodule_path = root_path + 'nodule/'
augment_nodule_path = root_path + 'augment_nodule/'
non_nodule_boxes_path = root_path + 'non_nodule_boxes/'

saving_path = root_path + 'much_data/'
if not os.path.exists(saving_path):
  os.makedirs(saving_path)

nodule_list = glob(nodule_path + "*.npy")
augment_nodule_list = glob(augment_nodule_path + "*.npy")
non_nodule_boxes_list = glob(non_nodule_boxes_path + "*.npy")

positive_label=np.array([0,1])
negative_label=np.array([1,0])

nodule_data = []
nodule_label = []

for nodule in nodule_list:
  img = np.load(nodule)
  img = np.reshape(img, crop_linear_len)
  nodule_data.append(img)
  nodule_label.append(positive_label)

nodule_data = np.array(nodule_data)
nodule_label = np.array(nodule_label)

np.save(saving_path + 'nodule_data.npy', nodule_data)
np.save(saving_path + 'nodule_label.npy', nodule_label)

augment_nodule_data = []
augment_nodule_label = []
for augment_nodule in augment_nodule_list:
  img = np.load(augment_nodule)
  img = np.reshape(img, crop_linear_len)
  augment_nodule_data.append(img)
  augment_nodule_label.append(positive_label)

augment_nodule_data = np.array(augment_nodule_data)
augment_nodule_label = np.array(augment_nodule_label)

np.save(saving_path + 'augment_nodule_data.npy', augment_nodule_data)
np.save(saving_path + 'augment_nodule_label.npy', augment_nodule_label)

non_nodule_boxes_data = []
non_nodule_boxes_label = []
for non_nodule_boxes in non_nodule_boxes_list:
  img = np.load(non_nodule_boxes)
  img = np.reshape(img, crop_linear_len)
  non_nodule_boxes_data.append(img)
  non_nodule_boxes_label.append(negative_label)

non_nodule_boxes_data = np.array(non_nodule_boxes_data)
non_nodule_boxes_label = np.array(non_nodule_boxes_label)

np.save(saving_path + 'non_nodule_boxes_data.npy', non_nodule_boxes_data)
np.save(saving_path + 'non_nodule_boxes_label.npy', non_nodule_boxes_label)

positive_data = np.append(nodule_data, augment_nodule_data, axis = 0)
positive_label = np.append(nodule_label, augment_nodule_label, axis = 0)

much_data = np.append(positive_data, non_nodule_boxes_data, axis = 0)
much_label = np.append(positive_label, non_nodule_boxes_label, axis = 0)

idx = np.array(range(len(much_data)))
random.shuffle(idx)
random.shuffle(idx)
much_data_shuffled = much_data[idx]
much_label_shuffled = much_label[idx]

np.save(saving_path + 'much_data_shuffled.npy', much_data_shuffled)
np.save(saving_path + 'much_label_shuffled.npy', much_label_shuffled)
 
