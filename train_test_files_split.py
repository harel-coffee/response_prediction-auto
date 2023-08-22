#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
#
# ---------------------------------------------------------------------------
# Created on Fri Feb  4 11:42:52 2022
#
# @author: SeyedM.MousaviKahaki (mousavikahaki@gmail.com)
#----------------------------------------------------------------------------
# Title:        Train Test Split
#
# Description:  This copy files into trainin and testing folders
#               
#
#
# Inputs:        CSV file: Training Set File
#                CSV file: Testing Set File
# Output:        None
#
# 
# Example:      train_test_files_split.py --src_dir INPUT_DIRECTORY --training_dir TRAINING_DIRECTORY --testing_dir TESTING_DIRECTORY --training_dataset TRAINING_DATASET --testing_dataset TESTING_DATASET
#               OR
#               runfile('train_test_files_split.py', args='--src_dir "C:/DATA/extracted_cutted_Augmented/data/png_files/" --training_dir "C:/DATA/extracted_cutted_Augmented/data/png_files/Training/" --testing_dir "C:/DATA/extracted_cutted_Augmented/data/png_files/Testing/" --training_dataset "C:/DATA/Aperio_TrainingSet_v10.csv" --testing_dataset "C:/DATA/Aperio_TestSet_v10.csv"')
#
#
# version ='3.0'
# ---------------------------------------------------------------------------

Created on Fri Feb  4 11:42:52 2022

@author: SeyedM.MousaviKahaki
"""


import os
import shutil
import pandas as pd
import glob
import argparse

    
    
    
##################### Separate Train and Test


def load(self, filename="config/param_spl.dat"):
    d = {"fname_field": "fname_field",
         "truth_field": "truth_field"
         }
    FILE = open(filename)
    for line in FILE:
        name, value = line.split(":")
        value = value.strip()
        if " " in value:
            value = map(str, value.split())
        else:
            value = str(value)
        setattr(self, d[name], value)
    FILE.close()

class A(object): pass
P = A()
load(P)
P.__dict__


parser = argparse.ArgumentParser(description='train test split')


parser.add_argument('--src_dir', type = str,
					help='path to the ourput directory')

parser.add_argument('--training_dir', type = str,
					help='name of the dataset')

parser.add_argument('--testing_dir', type = str,
					help='path to the dataset file')

parser.add_argument('--training_dataset', type = str,
					help='the dataset version')

parser.add_argument('--testing_dataset', type = str,
					help='the dataset version')


def main():
    
    args = parser.parse_args()
    
    src_dir = args.src_dir
    Training_dir = args.training_dir
    Test_dir = args.testing_dir
    
    TrainingDataset = args.training_dataset
    TestingDataset = args.testing_dataset
    
    Training = pd.read_csv(TrainingDataset)
    Testing = pd.read_csv(TestingDataset)
    
    filename_field = P.fname_field.replace('_' ,' ') # 'Filename of initial Aperio slide'
    # truth_field = P.truth_field # 'Responder'
    
    files = os.listdir(src_dir)
    for fname in files:
        src = src_dir+fname+'/'
        print(src)
        FName = fname.upper() + '.SVS'
        ExistTraining = Training[Training[filename_field] == FName].shape[0]
        ExistTesting = Testing[Testing[filename_field] == FName].shape[0]
        
        if ExistTraining:
            print("Training Set: " + src)
            # # All Images in one Folder
            # shutil.copytree(src, AE_Dir,dirs_exist_ok=True)
            shutil.move(src, Training_dir+fname)
            
        elif ExistTesting:
            print("Testing Set: " + src)
            shutil.move(src, Test_dir+fname)
        
    

if __name__ == '__main__':
    main()
    
    
   ## For Patch Scoring 
    
# src_dir =       'C:/DATA/2_extracted_cutted_Augmented/data_Outside/png_files/Training/'
# Training_dir =  'C:/DATA/Other/Patch-In-Out/train/'
# Test_dir =      'C:/DATA/Other/Patch-In-Out/test/'
# # AE_Dir =        'C:/DATA/extracted_cutted_Augmented/data/AE_Data/0/'

# files = os.listdir(src_dir)
# counter = 0
# for fname in files:
#     src = src_dir+fname+'/'
#     print(src)
    
#     patches_directory = glob.glob(r'C:/DATA/2_extracted_cutted_Augmented/data_Outside/png_files/Training/'+fname+'/*')
    
#     for filename in patches_directory:
#         print(filename)
#         targetName = 'Out.'+str(counter)+'.png'    
#         shutil.move(filename, Training_dir+targetName)
#         counter = counter + 1


# src_dir =       'C:/DATA/2_extracted_cutted_Augmented/data/png_files/Training/'  
# files = os.listdir(src_dir)
# counter = 0
# for fname in files:
#     src = src_dir+fname+'/'
#     print(src)
    
#     patches_directory = glob.glob(r'C:/DATA/2_extracted_cutted_Augmented/data/png_files/Training/'+fname+'/*')
    
#     for filename in patches_directory:
#         print(filename)
#         targetName = 'In.'+str(counter)+'.png'    
#         shutil.copy(filename, Training_dir+targetName)
#         counter = counter + 1    
    
    
    
    
    
    
    
    
    
    