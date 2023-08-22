#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
# ---------------------------------------------------------------------------
# Created on Fri Feb 11 10:17:47 2022
#
# @author: SeyedM.MousaviKahaki (mousavikahaki@gmail.com)
#----------------------------------------------------------------------------
# Title:        Patch Embeding
#
# Description:  This code perform patch embedding
#               There are three embedding models implimented in this project:
#               1. Autoencoder: an autoencoder model has been trained on training patches for embedding purpose
#               2. ResNet50: Pretrained ResNet50 trained on ImageNet dataset
#               3. VGG16: This model is loaded from keras application
#
# Input:        String: Source directory where the patch files are located
#               String: WEIGHTS_FOLDER which is the address to the pretrained model weights
#               String: OUTPUT_DIR the directory to save the extracted features
# Output:       Extracted Features per patch per WSI
#
# 
# Example:      embedding.py --source INPUT_DIR --save_dir OUTPUT_DIR --weight_folder WEIGHTS_FOLDER --embedding_model embedding_model
#               OR
#               runfile('embedding.py', args='--source "C:/DATA/2_extracted_cutted_Augmented/data/png_files" --save_dir "C:/DATA/Code/DigiPath_OWH/data/features" --weight_folder "models/saved_models/" --embedding_model "AE"')
#
# version ='3.0'
# ---------------------------------------------------------------------------

Created on Fri Feb 11 10:17:47 2022

@author: SeyedM.MousaviKahaki
"""


import numpy as np
from tensorflow.keras.optimizers import Adam, SGD
from keras.initializers import Orthogonal, HeUniform
import matplotlib.pyplot as plt
from keras.models import load_model
import os, os.path
import cv2
import glob
import pandas as pd
from tensorflow.python.keras.applications.resnet import ResNet50, preprocess_input
# from keras.preprocessing import image
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler, LabelEncoder
from keras.models import Sequential
from keras.layers import Dense
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, cohen_kappa_score
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import VarianceThreshold
from keras.layers import Input, Conv2D, Flatten, Dense, Conv2DTranspose, Reshape, Lambda, Activation, BatchNormalization, LeakyReLU, Dropout
from datetime import datetime
from keras.callbacks import EarlyStopping
from sklearn.metrics import roc_curve
from sklearn.metrics import auc
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_classif, f_regression
from sklearn.svm import LinearSVC
from sklearn.feature_selection import SelectFromModel
import pickle
import json
import h5py
import joblib
from scipy.stats import sem
from sklearn.metrics import roc_auc_score
from keras.applications.vgg16 import VGG16 
from keras.models import Model
from keras.applications.vgg16 import preprocess_input 
import argparse

# WEIGHTS_FOLDER = 'C:/DATA/Code/weights/'
# # DATA_FOLDER = 'C:/DATA/extracted_cutted/data/AE_Data/'
# # LOG_DIR = 'C:/DATA/Code/weights/logDir/'
# INPUT_DIR = 'C:/DATA/2_extracted_cutted_Augmented/data/png_files'
# OUTPUT_DIR = 'C:/DATA/Code/DigiPath_OWH/models/features'

def load(self, filename="config/param_emb.dat"):
    d = {"AE_model": "AE_model", 
         "DatasetFile": "DatasetFile",
         "fname_field": "fname_field",
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


parser = argparse.ArgumentParser(description='embedding')

parser.add_argument('--embedding_model', type = str,
					help='The embedding method: AE, ResNet50, or VGG16')

parser.add_argument('--source', type = str,
					help='path to folder containing image patches')

parser.add_argument('--weight_folder', type = str,
					help='path to folder containing model weights')
parser.add_argument('--save_dir', type = str,
					help='path to folder to save extracted features')



def main():
    # fo = open("C:/Users/SeyedM.MousaviKahaki/OneDrive - FDA/Documents/OWH_Project/WSI_Analysis/WSI/parameters.dat", "wb")
    # fo.close()
    args = parser.parse_args()
    INPUT_DIR = args.source
    WEIGHTS_FOLDER = args.weight_folder
    OUTPUT_DIR = args.save_dir
    embedding_model = args.embedding_model
    AE_model_name = P.AE_model#'epc200_im256_batch256_20220222-104322_EncoderModel.h5'
    DatasetFile = P.DatasetFile#'C:/DATA/Aperio_dataset_v10.csv'
    filename_field = P.fname_field.replace('_' ,' ') # 'Filename of initial Aperio slide'
    truth_field = P.truth_field # 'Responder'
    
    if embedding_model == 'AE':
        ############################################# Feature Extraction Using AE
        # simple_autoencoder_loaded = load_model(os.path.join(WEIGHTS_FOLDER, 'epc200_im256_batch256_20220222-104322_simple_autoencoderModel.h5'),
        #                                       compile=False) # epc200_im256_batch256_20220211-093252_simple_autoencoderModel.h5
        encoder_loaded = load_model(os.path.join(WEIGHTS_FOLDER, AE_model_name),
                                               compile=False) # epc200_im256_batch256_20220211-093252_EncoderModel.h5
    
        # DatasetFile = 'C:/DATA/Aperio_dataset_v7.csv'
        # DatasetFile = 'C:/DATA/Aperio_dataset_v10.csv'
        Dataset_ = pd.read_csv(DatasetFile)
    
        # Fixing Names having NEW
        # Dataset_['Filename of initial Aperio slide'] = Dataset_['Filename of initial Aperio slide'].replace('APERIO-045NEW-0.SVS' ,'APERIO-045-0.SVS')
        # Dataset_['Filename of initial Aperio slide'] = Dataset_['Filename of initial Aperio slide'].replace('APERIO-103NEW-0.SVS' ,'APERIO-103-0.SVS')
    
    
        ##### Training
        dirs = os.listdir(INPUT_DIR+'/Training/')
        allDataTrain = []
        labelTrain = []
        for dr in dirs:
            print("Processing: "+dr)
            FName = dr.upper() + '.SVS' 
            Responder = Dataset_[Dataset_[filename_field] == FName][truth_field].item()
            labelTrain.append(Responder)
            print("Responder: "+Responder)
            X_data = []
            files = glob.glob (INPUT_DIR+"/Training/"+dr+"/*.PNG")
            print(str(len(files))+ " File Found!")
            for myFile in files:
                image_ = cv2.imread(myFile)
                normalizedImg = cv2.normalize(image_, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
                X_data.append (normalizedImg)
            
            print('X_data shape:', np.array(X_data).shape)
            images_loaded = np.array(X_data)
            
            
            encodings_loaded = encoder_loaded.predict(images_loaded)
            # Save Features as HDF5
            with h5py.File(OUTPUT_DIR+'/AE/Training/'+dr+'.h5','w') as h5f:
                h5f.create_dataset("Features", data=np.asarray(encodings_loaded))
            
            print('encodings_loaded shape:', np.array(encodings_loaded).shape)
            
            allDataTrain.append(encodings_loaded)
    
        #### Save Training Lables as HDF5
        #### Create Binary Labels 
        lb = preprocessing.LabelBinarizer()
        y_train = lb.fit_transform(labelTrain)
    
        with h5py.File(OUTPUT_DIR+'/AE/TrainingLabels.h5','w') as h5f:
            h5f.create_dataset("Labels", data=np.asarray(y_train))
    
        ##### Tunning
        dirs = os.listdir(INPUT_DIR+'/Tunning/')
        allDataTunn = []
        labelTunn = []
        for dr in dirs:
            print("Processing: "+dr)
            FName = dr.upper() + '.SVS' 
            Responder = Dataset_[Dataset_[filename_field] == FName][truth_field].item()
            labelTunn.append(Responder)
            print("Responder: "+Responder)
            X_data = []
            files = glob.glob (INPUT_DIR+"/Tunning/"+dr+"/*.PNG")
            print(str(len(files))+ " File Found!")
            for myFile in files:
                image_ = cv2.imread(myFile)
                normalizedImg = cv2.normalize(image_, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
                X_data.append (normalizedImg)
            
            print('X_data shape:', np.array(X_data).shape)
            images_loaded = np.array(X_data)
            
            
            encodings_loaded = encoder_loaded.predict(images_loaded)
            # Save Features as HDF5
            with h5py.File(OUTPUT_DIR+'/AE/Tunning/'+dr+'.h5','w') as h5f:
                h5f.create_dataset("Features", data=np.asarray(encodings_loaded))
            
            print('encodings_loaded shape:', np.array(encodings_loaded).shape)
            
            allDataTunn.append(encodings_loaded)
        
        #### Save Training Lables as HDF5
        #### Create Binary Labels 
        lb = preprocessing.LabelBinarizer()
        y_Tunn = lb.fit_transform(labelTunn)
        
        with h5py.File(OUTPUT_DIR+'/AE/TunningLabels.h5','w') as h5f:
            h5f.create_dataset("Labels", data=np.asarray(y_Tunn))
    
    
    
    
        ##### Testing
        dirs = os.listdir(INPUT_DIR+'/Testing/')
        allDataTest = []
        labelTest = []
        for dr in dirs:
            print("Processing: "+dr)
            FName = dr.upper() + '.SVS' 
            Responder = Dataset_[Dataset_[filename_field] == FName][truth_field].item()
            labelTest.append(Responder)
            print("Responder: "+Responder)
            X_data = []
            files = glob.glob (INPUT_DIR+"/Testing/"+dr+"/*.PNG")
            print(str(len(files))+ " File Found!")
            for myFile in files:
                image_ = cv2.imread(myFile)
                normalizedImg = cv2.normalize(image_, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
                X_data.append (normalizedImg)
            
            print('X_data shape:', np.array(X_data).shape)
            images_loaded = np.array(X_data)
            
            
            encodings_loaded = encoder_loaded.predict(images_loaded)
            # Save Features as HDF5
            with h5py.File(OUTPUT_DIR+'/AE/Testing/'+dr+'.h5','w') as h5f:
                h5f.create_dataset("Features", data=np.asarray(encodings_loaded) )
            print('encodings_loaded shape:', np.array(encodings_loaded).shape)
            
            allDataTest.append(encodings_loaded)
            
        #### Save Training Lables as HDF5
        #### Create Binary Labels 
        lb = preprocessing.LabelBinarizer()
        y_test = lb.fit_transform(labelTest)
    
        with h5py.File(OUTPUT_DIR+'/AE/TestingLabels.h5','w') as h5f:
            h5f.create_dataset("Labels", data=np.asarray(y_test))
    
        images_loaded = []
        X_data = []
        
    elif embedding_model == 'resnet50':
        
        ############################################# Feature Extraction Using RESNET50 (Transfer Learning)
    
        model1 = ResNet50(weights='imagenet', pooling="avg", include_top = False) 
    
    
        # DatasetFile = 'C:/DATA/Aperio_dataset_v7.csv'
        # DatasetFile = 'C:/DATA/Aperio_dataset_v10.csv'
        Dataset_ = pd.read_csv(DatasetFile)
    
        # Fixing Names having NEW
        # Dataset_['Filename of initial Aperio slide'] = Dataset_['Filename of initial Aperio slide'].replace('APERIO-045NEW-0.SVS' ,'APERIO-045-0.SVS')
        # Dataset_['Filename of initial Aperio slide'] = Dataset_['Filename of initial Aperio slide'].replace('APERIO-103NEW-0.SVS' ,'APERIO-103-0.SVS')
    
    
        ##### Training
        dirs = os.listdir(INPUT_DIR+'/Training/')
        allDataTrain = []
        labelTrain = []
        for dr in dirs:
            print("Processing: "+dr)
            FName = dr.upper() + '.SVS' 
            Responder = Dataset_[Dataset_[filename_field] == FName][truth_field].item()
            labelTrain.append(Responder)
            print("Responder: "+Responder)
            X_data = []
            files = glob.glob (INPUT_DIR+"/Training/"+dr+"/*.PNG")
            print(str(len(files))+ " File Found!")
            for myFile in files:
                image_ = cv2.imread(myFile)
                normalizedImg = cv2.normalize(image_, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
                X_data.append (normalizedImg)
            
            print('X_data shape:', np.array(X_data).shape)
            ### Limit 200 Images for test ###################### TO REMOVE
            # X_data = X_data[0:400]
            images_loaded = np.array(X_data)
            
            
            encodings_loaded = model1.predict(images_loaded) 
            # Save Features as HDF5
            with h5py.File(OUTPUT_DIR+'/ResNet50/Training/'+dr+'.h5','w') as h5f:
                h5f.create_dataset("Features", data=np.asarray(encodings_loaded))
                
            print('encodings_loaded shape:', np.array(encodings_loaded).shape)
            
            allDataTrain.append(encodings_loaded)
    
        #### Save Training Lables as HDF5
        #### Create Binary Labels 
        lb = preprocessing.LabelBinarizer()
        y_train = lb.fit_transform(labelTrain)
    
        with h5py.File(OUTPUT_DIR+'/ResNet50/TrainingLabels.h5','w') as h5f:
            h5f.create_dataset("Labels", data=np.asarray(y_train))
            
        ##### Tunning
        dirs = os.listdir(INPUT_DIR+'/Tunning/')
        allDataTunn = []
        labelTunn = []
        for dr in dirs:
            print("Processing: "+dr)
            FName = dr.upper() + '.SVS' 
            Responder = Dataset_[Dataset_[filename_field] == FName][truth_field].item()
            labelTunn.append(Responder)
            print("Responder: "+Responder)
            X_data = []
            files = glob.glob (INPUT_DIR+"/Tunning/"+dr+"/*.PNG")
            print(str(len(files))+ " File Found!")
            for myFile in files:
                image_ = cv2.imread(myFile)
                normalizedImg = cv2.normalize(image_, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
                X_data.append (normalizedImg)
            
            print('X_data shape:', np.array(X_data).shape)
            ### Limit 200 Images for test ###################### TO REMOVE
            # X_data = X_data[0:400]
            images_loaded = np.array(X_data)
            
            
            encodings_loaded = model1.predict(images_loaded) 
            # Save Features as HDF5
            with h5py.File(OUTPUT_DIR+'/ResNet50/Tunning/'+dr+'.h5','w') as h5f:
                h5f.create_dataset("Features", data=np.asarray(encodings_loaded))
                
            print('encodings_loaded shape:', np.array(encodings_loaded).shape)
            
            allDataTunn.append(encodings_loaded)
        
        #### Save Tunning Lables as HDF5
        #### Create Binary Labels 
        lb = preprocessing.LabelBinarizer()
        y_Tunn = lb.fit_transform(labelTunn)
        
        with h5py.File(OUTPUT_DIR+'/ResNet50/TunningLabels.h5','w') as h5f:
            h5f.create_dataset("Labels", data=np.asarray(y_Tunn))
    
    
        ##### Testing
        dirs = os.listdir(INPUT_DIR+'/Testing/')
        allDataTest = []
        labelTest = []
        for dr in dirs:
            print("Processing: "+dr)
            FName = dr.upper() + '.SVS' 
            Responder = Dataset_[Dataset_[filename_field] == FName][truth_field].item()
            labelTest.append(Responder)
            print("Responder: "+Responder)
            X_data = []
            files = glob.glob (INPUT_DIR+"/Testing/"+dr+"/*.PNG")
            print(str(len(files))+ " File Found!")
            for myFile in files:
                image_ = cv2.imread(myFile)
                normalizedImg = cv2.normalize(image_, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
                X_data.append (normalizedImg)
            
            print('X_data shape:', np.array(X_data).shape)
            ### Limit 200 Images for test ###################### TO REMOVE
            # X_data = X_data[0:400]
            images_loaded = np.array(X_data)
            
            
            encodings_loaded = model1.predict(images_loaded)
            # Save Features as HDF5
            with h5py.File(OUTPUT_DIR+'/ResNet50/Testing/'+dr+'.h5','w') as h5f:
                h5f.create_dataset("Features", data=np.asarray(encodings_loaded))
            print('encodings_loaded shape:', np.array(encodings_loaded).shape)
            
            allDataTest.append(encodings_loaded)
    
        #### Save Training Lables as HDF5
        #### Create Binary Labels 
        lb = preprocessing.LabelBinarizer()
        y_test = lb.fit_transform(labelTest)
    
        with h5py.File(OUTPUT_DIR+'/ResNet50/TestingLabels.h5','w') as h5f:
            h5f.create_dataset("Labels", data=np.asarray(y_test))
    
        images_loaded = []
        X_data = []
    
        images_loaded = []
        X_data = []
        
        
    elif embedding_model == 'vgg16':
        
        ############################################# Feature Extraction Using VGG16 (Transfer Learning)
    
        # model1 = ResNet50(weights='imagenet', pooling="avg", include_top = False) 
    
        model = VGG16()
        model = Model(inputs = model.inputs, outputs = model.layers[-2].output)
    
        # DatasetFile = 'C:/DATA/Aperio_dataset_v7.csv'
        # DatasetFile = 'C:/DATA/Aperio_dataset_v10.csv'
        Dataset_ = pd.read_csv(DatasetFile)
    
        # Fixing Names having NEW
        # Dataset_['Filename of initial Aperio slide'] = Dataset_['Filename of initial Aperio slide'].replace('APERIO-045NEW-0.SVS' ,'APERIO-045-0.SVS')
        # Dataset_['Filename of initial Aperio slide'] = Dataset_['Filename of initial Aperio slide'].replace('APERIO-103NEW-0.SVS' ,'APERIO-103-0.SVS')
    
    
        ##### Training
        dirs = os.listdir(INPUT_DIR+'/Training/')
        allDataTrain = []
        labelTrain = []
        for dr in dirs:
            print("Processing: "+dr)
            FName = dr.upper() + '.SVS' 
            Responder = Dataset_[Dataset_[filename_field] == FName][truth_field].item()
            labelTrain.append(Responder)
            print("Responder: "+Responder)
            X_data = []
            files = glob.glob (INPUT_DIR+"/Training/"+dr+"/*.PNG")
            print(str(len(files))+ " File Found!")
            for myFile in files:
                image_ = cv2.imread(myFile)
                # for VGG16
                dim = (224, 224)
                # resize image
                image_ = cv2.resize(image_, dim, interpolation = cv2.INTER_AREA)
                
                # # prepare image for model - Try this
                # imgx = preprocess_input(reshaped_img)
                
                normalizedImg = cv2.normalize(image_, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
                X_data.append (normalizedImg)
            
            print('X_data shape:', np.array(X_data).shape)
            ### Limit 200 Images for test ###################### TO REMOVE
            # X_data = X_data[0:400]
            images_loaded = np.array(X_data)
            
            
            # encodings_loaded = model1.predict(images_loaded) 
            encodings_loaded = model.predict(images_loaded) 
            # Save Features as HDF5
            with h5py.File(OUTPUT_DIR+'/VGG16/Training/'+dr+'.h5','w') as h5f:
                h5f.create_dataset("Features", data=np.asarray(encodings_loaded))
                
            print('encodings_loaded shape:', np.array(encodings_loaded).shape)
            
            allDataTrain.append(encodings_loaded)
    
        #### Save Training Lables as HDF5
        #### Create Binary Labels 
        lb = preprocessing.LabelBinarizer()
        y_train = lb.fit_transform(labelTrain)
    
        with h5py.File(OUTPUT_DIR+'/VGG16/TrainingLabels.h5','w') as h5f:
            h5f.create_dataset("Labels", data=np.asarray(y_train))
            
        ##### Testing
        dirs = os.listdir(INPUT_DIR+'/Testing/')
        allDataTest = []
        labelTest = []
        for dr in dirs:
            print("Processing: "+dr)
            FName = dr.upper() + '.SVS' 
            Responder = Dataset_[Dataset_[filename_field] == FName][truth_field].item()
            labelTest.append(Responder)
            print("Responder: "+Responder)
            X_data = []
            files = glob.glob (INPUT_DIR+"/Testing/"+dr+"/*.PNG")
            print(str(len(files))+ " File Found!")
            for myFile in files:
                image_ = cv2.imread(myFile)
                # for VGG16
                dim = (224, 224)
                # resize image
                image_ = cv2.resize(image_, dim, interpolation = cv2.INTER_AREA)
                
                normalizedImg = cv2.normalize(image_, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
                X_data.append (normalizedImg)
            
            print('X_data shape:', np.array(X_data).shape)
            ### Limit 200 Images for test ###################### TO REMOVE
            # X_data = X_data[0:400]
            images_loaded = np.array(X_data)
            
            
            encodings_loaded = model.predict(images_loaded)
            # Save Features as HDF5
            with h5py.File(OUTPUT_DIR+'/VGG16/Testing/'+dr+'.h5','w') as h5f:
                h5f.create_dataset("Features", data=np.asarray(encodings_loaded))
            print('encodings_loaded shape:', np.array(encodings_loaded).shape)
            
            allDataTest.append(encodings_loaded)
    
        #### Save Training Lables as HDF5
        #### Create Binary Labels 
        lb = preprocessing.LabelBinarizer()
        y_test = lb.fit_transform(labelTest)
    
        with h5py.File(OUTPUT_DIR+'/VGG16/TestingLabels.h5','w') as h5f:
            h5f.create_dataset("Labels", data=np.asarray(y_test))
    
        images_loaded = []
        X_data = []
    
        images_loaded = []
        X_data = []
        
        
    elif embedding_model == 'GoogleConstructiveLearning':
        
            print("NotImplementedError")



if __name__ == '__main__':
    main()




