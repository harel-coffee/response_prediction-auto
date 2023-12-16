#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# ---------------------------------------------------------------------------
# Created on Wed Sep 29 10:35:19 2021
#
# @author: SeyedM.MousaviKahaki (mousavikahaki@gmail.com)
#----------------------------------------------------------------------------
# Title:        Annotation Augmentation
#
# Description:  This code augment annotations to have the same number of extracted annotations per WSI
#               
#
#
# Input:        String: Source directory where the extracted annotations are located   
# Output:       Augmented annotations
#
# 
# Example:      augmentation.py
#
#
# version ='3.0'
# ---------------------------------------------------------------------------
# -*- coding: utf-8 -*-
"""
Created on Sun Feb 20 10:03:36 2022

@author: SeyedM.MousaviKahaki
"""

import os
import shutil
import pandas as pd
import cv2
import matplotlib.pyplot as plt
import numpy as np
from scipy import ndimage



src_dir = 'C:/DATA/Oklahoma_Extracted_New_FixedCircle_processed_Augmented_combined/'
dest_dir = 'C:/DATA/Oklahoma_Extracted_New_FixedCircle_processed_Augmented_combined/'


# #### Remove Benign Area
# folders = os.listdir(src_dir)
# substring = 'Benign'
# for foldername in folders:
#     src = src_dir+foldername+'/'
#     print(src)
#     filenames = next(os.walk(src))[2]
#     for fname in filenames:
#         if substring in fname:
#             print("Removing " + src+ fname)
#             os.remove(src+ fname)

                
Augm_list = ['90','180','h','v','vh','90v','90h','90vh','180v','180h','180vh',
             '90','180','h','v','vh','90v','90h','90vh','180v','180h','180vh',
             '90','180','h','v','vh','90v','90h','90vh','180v','180h','180vh']    
    
    
DatasetFile = 'C:/DATA/Aperio_dataset_v9.csv'
Dataset_ = pd.read_csv(DatasetFile)    
    
    
min(Dataset_['Num CAH']+Dataset_['Num Carcinoma'])
max_annot = max(Dataset_['Num CAH']+Dataset_['Num Carcinoma'])
# max_annot = 37

#### Augmenting to Max # of annotation


def Do_Aug(OriginalFname,NewFname,Augm):
    image_ = cv2.imread(OriginalFname)
    # plt.imshow(image_) 
    # plt.title('Original')
    # plt.axis('off')
    # plt.show()
    if Augm == 'h':
        imageN = cv2.flip(image_, 0)
    if Augm == 'v':
        imageN = cv2.flip(image_, 1)
    if Augm == 'vh':
        imageN = cv2.flip(image_, -1)
    if Augm == '90':
        imageN = cv2.rotate(image_, cv2.cv2.ROTATE_90_CLOCKWISE) # cv2.cv2.ROTATE_90_COUNTERCLOCKWISE
    if Augm == '90h':
        imageN = cv2.rotate(image_, cv2.cv2.ROTATE_90_CLOCKWISE)
        imageN = cv2.flip(imageN, 0)
    if Augm == '90v':
        imageN = cv2.rotate(image_, cv2.cv2.ROTATE_90_CLOCKWISE)
        imageN = cv2.flip(imageN, 1)
    if Augm == '90vh':
        imageN = cv2.rotate(image_, cv2.cv2.ROTATE_90_CLOCKWISE)
        imageN = cv2.flip(imageN, -1)  
    if Augm == '180':
        imageN = cv2.rotate(image_, cv2.cv2.ROTATE_180)
    if Augm == '180h':
        imageN = cv2.rotate(image_, cv2.cv2.ROTATE_180)
        imageN = cv2.flip(imageN, 0)
    if Augm == '180v':
        imageN = cv2.rotate(image_, cv2.cv2.ROTATE_180)
        imageN = cv2.flip(imageN, 1)
    if Augm == '180vh':
        imageN = cv2.rotate(image_, cv2.cv2.ROTATE_180)
        imageN = cv2.flip(imageN, -1)
    # plt.imshow(imageN) 
    # plt.title(Augm)
    # plt.axis('off')
    # plt.show()
    cv2.imwrite(NewFname, imageN)     
 

folders = os.listdir(src_dir)

for foldername in folders:
    src = src_dir+foldername+'/'
    print(src)
    filenames = next(os.walk(src))[2]
    augm_nedded = max_annot - len(filenames)
    
    counter = augm_nedded
    
    for Augm in Augm_list:
        for fname in filenames: 
            OriginalFname = src+fname
            print('File Name: ' + fname)
            print('Counter: ' + str(counter))
            if counter > 0:
                parts = fname.split('.')
                NewFname = src+parts[0]+'_Aug_'+Augm + '.'+ parts[1]
                
                Do_Aug(OriginalFname,NewFname,Augm)
                print('Augmented:' + NewFname)
                counter = counter - 1