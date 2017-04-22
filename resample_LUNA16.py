from glob import glob

import scipy.ndimage
import SimpleITK as sitk
import numpy as np
import os

# multi threads
from joblib import Parallel, delayed
import multiprocessing

## Read annotation data
# Learned from Jonathan Mulholland and Aaron Sander, Booz Allen Hamilton
# https://www.kaggle.com/c/data-science-bowl-2017#tutorial

luna_path = '/LUNA16/Ori_Data/'
luna_subset_path = luna_path + 'subset*/'
file_list = glob(luna_subset_path + "*.mhd")


## Define resample method to make images isomorphic, default spacing is [1, 1, 1]mm
# Learned from Guido Zuidhof
# https://www.kaggle.com/gzuidhof/data-science-bowl-2017/full-preprocessing-tutorial
def resample(image, old_spacing, new_spacing=[1, 1, 1]):

    resize_factor = old_spacing / new_spacing
    new_real_shape = image.shape * resize_factor
    new_shape = np.round(new_real_shape)
    real_resize_factor = new_shape / image.shape
    new_spacing = old_spacing / real_resize_factor

    image = scipy.ndimage.interpolation.zoom(image, real_resize_factor, mode = 'nearest')

    return image, new_spacing


def resample_for_patient(patient, save_root_path = '/LUNA16/Resampled/'):
    # Read CT scan
    full_image_info = sitk.ReadImage(patient)
    full_scan = sitk.GetArrayFromImage(full_image_info)

    # Get origin and spacing
    origin = np.array(full_image_info.GetOrigin())[::-1] # get [z, y, x] origin
    old_spacing = np.array(full_image_info.GetSpacing())[::-1] # get [z, y, x] spacing

    # Resample and save
    image, new_spacing = resample(full_scan, old_spacing)
    patient_name = str(os.path.split(patient)[1]).replace('.mhd', '')

    save_path = save_root_path + patient_name

    np.save(save_path + '_resampled.npy', image)
    np.save(save_path + '_origin.npy', origin)
    np.save(save_path + '_new_spacing.npy', new_spacing)

num_cores = multiprocessing.cpu_count()
Parallel(n_jobs = num_cores)(delayed(resample_for_patient)(patient_path) for patient_path in file_list)

print('Done')
