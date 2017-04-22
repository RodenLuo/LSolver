# coding: utf-8
import sys
import numpy as np # linear algebra

subset = sys.argv[1]
crop_window_len = np.int(sys.argv[2])

saving_mm_name = str(crop_window_len * 2 +1) + 'mm'

import cv2
from skimage import segmentation
from sklearn.cluster import DBSCAN


import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import skimage, os
from skimage.morphology import ball, disk, dilation, binary_erosion, remove_small_objects, erosion, closing, reconstruction, binary_closing
from skimage.measure import label,regionprops, perimeter
from skimage.morphology import binary_dilation, binary_opening
from skimage.filters import roberts, sobel
from skimage import measure, feature
from skimage.segmentation import clear_border
# from skimage import data
from scipy import ndimage as ndi
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import dicom
import scipy.misc
import numpy as np
from skimage.segmentation import clear_border
from skimage.feature import peak_local_max
from scipy.ndimage.interpolation import rotate 

import os
import numpy
import array

def save_nodule(nodule_crop, name_, path):
    np.save(path + str(name_) + '.npy', nodule_crop)

import SimpleITK as sitk
import numpy as np

from glob import glob
import pandas as pd
import scipy.ndimage

## Read annotation data and filter those without images
# Learned from Jonathan Mulholland and Aaron Sander, Booz Allen Hamilton
# https://www.kaggle.com/c/data-science-bowl-2017#tutorial


# Predefine some parameters, this will affect final performance
low_cutoff = -650
jitter_number = 120
rotate_number = 360; num_per_direction = np.int(rotate_number/6);


# Set input path
nodule_path = '/LUNA16/Train_data/' + saving_mm_name + '/' + subset + '/nodule/'
nodule_list = glob(nodule_path + "*.npy")

saving_path = '/LUNA16/Train_data/' + saving_mm_name + '/' + subset + '/augment_nodule/'
if not os.path.exists(saving_path):
    os.makedirs(saving_path) 
    

def augment_nodule(nodule_npy_path):
    
    nodule_crop = np.load(nodule_npy_path)
    
    nodule_name = str(os.path.split(nodule_npy_path)[1]).replace('.npy', '')

    i = 0
    for ax_1 in range(3):
        for ax_2 in range(3):
            if not ax_2 == ax_1:
                random_angle = np.random.uniform(0, 360, num_per_direction)
                for angle in random_angle:
                    i+=1    
                    nodule_crop_r = rotate(nodule_crop, angle, axes=(ax_1, ax_2), reshape=False, mode='reflect')
                    save_nodule(nodule_crop_r, nodule_name + '_rotate_' + str(i), saving_path)

    for idx in range(jitter_number):
        random_jitter = np.rint(np.random.uniform(-5, 5, [crop_window_len * 2 +1, crop_window_len * 2 +1, crop_window_len * 2 +1]))
        nodule_crop_j = np.array(nodule_crop + random_jitter, dtype=np.int16)
        save_nodule(nodule_crop_j, nodule_name + '_jitter_' + str(idx), saving_path)
        
        
from joblib import Parallel, delayed
import multiprocessing
num_cores = multiprocessing.cpu_count()

Parallel(n_jobs=num_cores)(delayed(augment_nodule)(nodule_path) for nodule_path in nodule_list)

print('Done for all')
