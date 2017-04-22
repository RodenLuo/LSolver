# coding: utf-8
import sys
import numpy as np # linear algebra

subset = sys.argv[1]
crop_window_len = np.int(sys.argv[2])

# subset = 'train1'
# crop_window_len = 13

saving_mm_name = str(crop_window_len * 2 +1) + 'mm_POI'

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
#!/usr/bin/env python


#======================================================================
#Program:   Diffusion Weighted MRI Reconstruction
#Link:      https://code.google.com/archive/p/diffusion-mri
#Module:    $RCSfile: mhd_utils.py,v $
#Language:  Python
#Author:    $Author: bjian $
#Date:      $Date: 2008/10/27 05:55:55 $
#Version:
#           $Revision: 1.1 by PJackson 2013/06/06 $
#               Modification: Adapted to 3D
#               Link: https://sites.google.com/site/pjmedphys/tutorials/medical-images-in-python
#
#           $Revision: 2   by RodenLuo 2017/03/12 $
#               Modication: Adapted to LUNA2016 data set for DSB2017
#               Link:
#======================================================================

import os
import numpy
import array

def save_nodule(nodule_crop, name_index, path):
    np.save(path + str(name_index) + '.npy', nodule_crop)
    # write_mhd_file(path + str(name_index) + '.mhd', nodule_crop, nodule_crop.shape[::-1])

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
subsample_number = 200

# Set input path
luna_path = '/LUNA16/Ori_data/'
df_candidate = pd.read_csv(luna_path+'CSVFILES/candidates_V2_w_file_path.csv')

luna_mhd_path = luna_path + subset +'/'
file_list = glob(luna_mhd_path + "*.mhd")

def crop_nodule(big_img, v_center, crop_len = crop_window_len):
    '''
    img and v_center is in [Z, Y, X], and in numpy.array type
    '''
    [max_Z, max_Y, max_X] = big_img.shape

    zyx_1 = v_center - crop_len # Attention: Z, Y, X
    zyx_2 = v_center + crop_len + 1

    img_crop = big_img[ max(zyx_1[0], 0):min(zyx_2[0], max_Z),
                       max(zyx_1[1], 0):min(zyx_2[1], max_Y),
                       max(zyx_1[2], 0):min(zyx_2[2], max_X) ]

    [crop_Z, crop_Y, crop_X] = img_crop.shape
    if min(crop_Z, crop_Y, crop_X) < crop_len*2+1:
        crop_block_len = crop_len*2+1

        crop_block = np.array(np.ones([crop_block_len, crop_block_len, crop_block_len]))
        crop_block *= -1024
        crop_block = np.asarray(crop_block, dtype=np.int16)

        start_Z = int((crop_block_len - crop_Z)/2)
        start_Y = int((crop_block_len - crop_Y)/2)
        start_X = int((crop_block_len - crop_X)/2)

        crop_block[start_Z:start_Z+crop_Z, start_Y:start_Y+crop_Y, start_X:start_X+crop_X] = img_crop

        img_crop = crop_block

    return img_crop

# http://stackoverflow.com/questions/10818546/finding-index-of-nearest-point-in-numpy-arrays-of-x-and-y-coordinates
# http://stackoverflow.com/questions/32424604/find-all-nearest-neighbors-within-a-specific-distance

from scipy import spatial
## Collect patients with nodule and crop the nodule
# In this code snippet, the cropped nodule is a [19, 19, 19] volume with [1, 1, 1]mm spacing.
# Learned from Jonathan Mulholland and Aaron Sander, Booz Allen Hamilton
# https://www.kaggle.com/c/data-science-bowl-2017#tutorial

def grids_generaotr(img_shape, grid_size = 30):

    grid_shape = img_shape
    grids = []

    start_ind = [0, np.int(grid_size/2)]

    for Z_ in range(len(start_ind)):
        for Y_ in range(len(start_ind)):
            for X_ in range(len(start_ind)):

                # initialize
                grid = np.array(np.zeros(grid_shape))
                indices=[None] * 3

                # get grid indices
                indices[0] = np.array(range(grid_shape[0]))[start_ind[Z_]::grid_size]
                indices[1] = np.array(range(grid_shape[1]))[start_ind[Y_]::grid_size]
                indices[2] = np.array(range(grid_shape[2]))[start_ind[X_]::grid_size]

                # create grid from indices
                grid[indices[0], :, :] = 1
                grid[:, indices[1], :] = 1
                grid[:, :, indices[2]] = 1

                grid = grid > 0

                # add to list
                grids.append(grid)
                
    return grids


def seg_one_slice_seg(img, large_label_size = 3, low_cutoff_ = -650):
    img_bw = img > low_cutoff_
#     img_bw = img_bw * 255
    img_bw = np.array(img_bw, dtype=np.uint8)
    
    img_label = label(img_bw)
    img_label_props = regionprops(img_label)
    
    large_label = np.array(np.zeros(img.shape), dtype=np.uint8)

    for r in img_label_props:
        max_x, max_y = 0, 0
        min_x, min_y = 1000, 1000
    
        for c in r.coords:
            max_y = max(c[0], max_y)
            max_x = max(c[1], max_x)

            min_y = min(c[0], min_y)
            min_x = min(c[1], min_x)
            
        if ( (max_y - min_y) < 3 or (max_x - min_x) < 3 ):
            for c in r.coords:
                img_bw[ c[0], c[1] ] = 0
    
    # Area threshold
    img_label = label(img_bw)
    img_label_props = regionprops(img_label)
    
    for r in img_label_props:
        if ( r.area > large_label_size ):
            for c in r.coords:
                large_label[c[0], c[1]] = 255        
    
    # Finding sure foreground area for large label
    dist_transform = cv2.distanceTransform(large_label, cv2.DIST_L2, 5)
    distance_threshold = 1.0
    ret, large_label_processed = cv2.threshold(dist_transform, distance_threshold, 255, 0)

    return large_label_processed


def break_large_label_by_grid(large_label, grid):
    large_label_break = np.array(large_label)
    large_label_break[grid] = 0
    break_centroid = np.array([r.centroid for r in regionprops(label(large_label_break))])
    
    return break_centroid


def collect_POI_from_lung(lung_image, large_diameter_threshold = 4):
    processed_lung = np.array([seg_one_slice_seg(img) for img in lung_image])
    processed_lung_label_props = regionprops(label(processed_lung))

    small_label = np.array(np.zeros(processed_lung.shape), dtype=np.uint8)
    large_label = np.array(np.zeros(processed_lung.shape), dtype=np.uint8)

    for r in processed_lung_label_props:
        if ( r.equivalent_diameter > large_diameter_threshold ):
            for c in r.coords:
                large_label[c[0], c[1], c[2]] = 255
        else: 
            for c in r.coords:
                small_label[c[0], c[1], c[2]] = 255

    all_centroid = np.array([r.centroid for r in regionprops(label(small_label))])

    grids = grids_generaotr(large_label.shape, grid_size = 30)
    for grid in grids:
        break_centroid = break_large_label_by_grid(large_label, grid)
        all_centroid = np.append(all_centroid, break_centroid, axis = 0)
        
    return all_centroid


def get_POI(ct_lung):
    all_centroid = collect_POI_from_lung(ct_lung)
    db = DBSCAN(eps=1, min_samples=1).fit(all_centroid)
    all_POI = np.array([np.mean(all_centroid[db.labels_ == ind], axis=0) for ind in np.unique(db.labels_)])
    
    
    return np.array(np.rint(all_POI), dtype=np.int)



def get_nodule_for_patient(patient):

    # Check whether this patient has nodule or not
    patient_nodules = df_node[df_node.file == patient]

    ct_scan = np.load(patient)

    patient_name = str(os.path.split(patient)[1]).replace('.npy', '')
    
    ct_lung = np.load('/DSB2017/LUNA16/new_lung/lung/' + patient_name + '_lung.npy')
    
    all_POI = get_POI(ct_lung)
    
    # load metadata
    origin = np.load('/DSB2017/LUNA16/resampled_origin_new_spacing_npy/' + patient_name + '_origin.npy')
    new_spacing = np.load('/DSB2017/LUNA16/resampled_origin_new_spacing_npy/' + patient_name + '_new_spacing.npy')

    
    
#     print('Start save nodule')
    for index, nodule in patient_nodules.iterrows():
        if nodule.diameter_mm < 4:
            print('Patient:' + patient + ', ' + 'Nodule: ' + str(index) + ' is too small, diameter: ' + str(nodule.diameter_mm))
            continue 
        nodule_center = np.array([nodule.coordZ, nodule.coordY, nodule.coordX]) # Attention: Z, Y, X

        v_center = np.rint( (nodule_center - origin) / new_spacing ).astype(int)
        
        # find POI closest to nodule center
        POI_idx_closest_to_center = spatial.KDTree(all_POI).query(v_center)[1]

        center_POI = all_POI[POI_idx_closest_to_center]
        dist = numpy.linalg.norm(center_POI - v_center)
        if dist > 9:
            print('Patient:' + patient + ', ' + 'Nodule: ' + str(index) +
                  ', True_Center: ' + np.array_str(v_center) + ', CenterPOI: ' +
                  np.array_str(center_POI) + ', Distance: '+ str(dist) +' is too long')
            return # do not consider the whole patient

        # crop and save nodule
        img_crop = crop_nodule(ct_scan, v_center=np.array(center_POI))
        saving_path = '/Train_data/' + saving_mm_name + '/'+subset+'/nodule/'
    
        if not os.path.exists(saving_path):
            os.makedirs(saving_path) 
        save_nodule(img_crop, index, path = saving_path)
        print('Patient:' + patient + ', ' + 'Nodule: ' + str(index) +
                  ', True_Center: ' + np.array_str(v_center) + ', CenterPOI: ' +
                  np.array_str(center_POI) + ', Distance: '+ str(dist))
        
        
        # remove the surrounding POIs
        point_tree = spatial.cKDTree(all_POI)
        with_in_range_index = point_tree.query_ball_point(center_POI, crop_window_len)
        all_POI = np.delete(all_POI, with_in_range_index, 0)
           

    # crop non-nodule boxes
        
    new_POI_img = np.zeros(ct_lung.shape)
    new_POI_img[ all_POI[:,0], all_POI[:,1], all_POI[:,2] ] = 1
    
    new_all_POI = np.argwhere(new_POI_img>0)
    new_all_POI_subsample = new_all_POI[
        np.random.choice(len(new_all_POI), subsample_number, replace = False)]
    
    for coor in new_all_POI_subsample:
        img_crop = crop_nodule(ct_scan, v_center=np.array(coor))

        save_name = patient_name+'_'+str(coor[0])+'_'+str(coor[1])+'_'+str(coor[2])
        
        saving_path = '/Train_data/' + saving_mm_name + '/'+subset+'/non_nodule_boxes/'
        if not os.path.exists(saving_path):
            os.makedirs(saving_path) 

        save_nodule(img_crop, save_name, path = saving_path)
        

from joblib import Parallel, delayed
import multiprocessing
num_cores = multiprocessing.cpu_count()

Parallel(n_jobs=num_cores)(delayed(get_nodule_for_patient)(patient_name) for patient_name in file_list)

print('Done for all')
