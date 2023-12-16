#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
#
# ---------------------------------------------------------------------------
# Created on Fri Feb 11 10:17:47 2022
#
# @author: SeyedM.MousaviKahaki (mousavikahaki@gmail.com)
#----------------------------------------------------------------------------
# Title:        training Resnet50
#
# Description:  This code import the embedded features, define a model, and perform training and validation
#
# Input:        String: Source directory where the WSIs and corresponding XML files are located  
# Output:       Extracted annotations
#
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
import argparse

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

def load(self, filename="config/parameters.dat"):
    d = {"k": "k", "Fn": "Fn", "batchSize1": "batchSize1",
         "nepochs": "nepochs","validationSplit": "validationSplit","lr1": "lr1",
         "decay": "decay","momentum": "momentum","nesterov": "nesterov",
         "beta_1": "beta_1","beta_2": "beta_2",
         "epsilon": "epsilon","patience": "patience","verbose": "verbose",
         "Dropout1": "Dropout1","n_bootstraps": "n_bootstraps",
         "rng_seed": "rng_seed","seeds1": "seeds1"
         }
    FILE = open(filename)
    for line in FILE:
        name, value = line.split(":")
        value = value.strip()
        if " " in value:
            value = map(float, value.split())
        else:
            value = float(value)
        setattr(self, d[name], value)
    FILE.close()

class A(object): pass
a = A()
load(a)
a.__dict__

savemode = True


def _str_to_bool(s):
    """Convert string to bool (in argparse context)."""
    if s.lower() not in ['true', 'false']:
        raise ValueError('Need bool; got %r' % s)
    return {'true': True, 'false': False}[s.lower()]

def add_boolean_argument(parser, name, default=False):                                                                                               
    """Add a boolean argument to an ArgumentParser instance."""
    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        '--' + name, nargs='?', default=default, const=True, type=_str_to_bool)
    group.add_argument('--no' + name, dest=name, action='store_false')

parser = argparse.ArgumentParser(description='training autoencoder')

parser.add_argument('--weightes_folder', type = str,
					help='path to folder to load and save model weights')

parser.add_argument('--training_data', type = str,
					help='path to folder containing raw wsi image files')
parser.add_argument('--training_labels', type = str,
					help='path to folder to save extracted annotations')

parser.add_argument('--tunning_data', type = str,
					help='path to folder containing raw wsi image files')
parser.add_argument('--tunning_labels', type = str,
					help='path to folder to save extracted annotations')

parser.add_argument('--testing_data', type = str,
					help='path to folder containing raw wsi image files')
parser.add_argument('--testing_labels', type = str,
					help='path to folder to save extracted annotations')


add_boolean_argument(parser, 'random_forest', default=False)
add_boolean_argument(parser, 'bootsrap', default=False)


def embedding_ResNet50():
    ############################################# Feature Extraction Using RESNET50 (Transfer Learning)
    args = parser.parse_args()
    INPUT_DIR = args.source
    # WEIGHTS_FOLDER = args.weight_folder
    OUTPUT_DIR = args.save_dir

    DatasetFile = P.DatasetFile
    filename_field = P.fname_field.replace('_' ,' ') 
    truth_field = P.truth_field # 'Responder'
    
    model1 = ResNet50(weights='imagenet', pooling="avg", include_top = False) 


    Dataset_ = pd.read_csv(DatasetFile)

    ##### Training Data
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


def main():
    
    
    args = parser.parse_args()

    WEIGHTS_FOLDER = args.weightes_folder 
    training_data = args.training_data 
    training_labels = args.training_labels 
    
    tunning_data = args.tunning_data 
    tunning_labels = args.tunning_labels 
    

    
    #############################################  READ FEATURES From HDF5 Files
    

    dirs = os.listdir(training_data)
    allDataTrain = []
    y_train = h5py.File(training_labels)['Labels'][:]
    
    for dr in dirs:
        print("Processing: "+dr)
        # Read Features as HDF5   
        encodings_loaded = h5py.File(training_data+dr)['Features'][:]    
        allDataTrain.append(encodings_loaded)
    
    
    ######## Load Tunning Data
    dirs = os.listdir(tunning_data)
    allDataTunn = []

    y_Tunn = h5py.File(tunning_labels)['Labels'][:]
    
    for dr in dirs:
        print("Processing: "+dr)
        # Read Features as HDF5   
        encodings_loaded = h5py.File(tunning_data+dr)['Features'][:]    
        allDataTunn.append(encodings_loaded)
                
    

    
    ################# Create Training and Test Dataframes from Data
    
    X_trainO = []
    X_trainO = pd.DataFrame(X_trainO)
    for i in range(len(allDataTrain)):
        print(i)
        print(len(allDataTrain[i]))
        Adf=[]
        Adf = allDataTrain[i].mean(axis=0)
        #Adf = pd.DataFrame(allDataTrain[i]).values.flatten()
        X_trainO = pd.concat([X_trainO, pd.DataFrame(Adf).T], axis=0)
        
        
    X_tunnO = []
    X_tunnO = pd.DataFrame(X_tunnO)
    for i in range(len(allDataTunn)):
        print(i)
        print(len(allDataTunn[i]))
        Adf=[]
        Adf = allDataTunn[i].mean(axis=0)
        #Adf = pd.DataFrame(allDataTunn[i]).values.flatten()
        X_tunnO = pd.concat([X_tunnO, pd.DataFrame(Adf).T], axis=0)
    

    
    X_train = X_trainO
    X_tunn = X_tunnO
           
    y_train = y_train.astype('float')
    y_Tunn = y_Tunn.astype('float')
    
    X = X_train
    X = X.fillna(-1)
    
    X_tunn = X_tunn.fillna(-1)
    X = X.dropna(axis=1, how='any')
    X_tunn = X_tunn.dropna(axis=1, how='any')
    
    model = Sequential()    
    init2 = Orthogonal(seed = int(a.seeds1))

    model.add(Dense(128, input_dim=X.shape[1], kernel_initializer=init2)) 
    model.add(BatchNormalization())
    model.add(Activation('relu'))#tanh relu
    model.add(Dropout(a.Dropout1))
    
 
    model.add(Dense(64, kernel_initializer=init2))
    model.add(BatchNormalization())
    model.add(Activation('relu'))#tanh relu
    model.add(Dropout(a.Dropout1))

    model.add(Dense(32, kernel_initializer=init2))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dropout(a.Dropout1))
    
    model.add(Dense(1, kernel_initializer=init2))
    model.add(BatchNormalization())
    model.add(Activation('sigmoid'))
    
    # setting up the optimization of our weights 
    sgd = SGD(lr=a.lr1, decay=1e-6, momentum=a.momentum, nesterov=True)
    adm = Adam(lr=a.lr1, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=1e-6)
    model.compile(loss='binary_crossentropy', optimizer=adm, metrics=['accuracy']) # squared_hinge
    
    batchSize = int(a.batchSize)
    nepochs = int(a.nepochs)
    validationSplit = a.validationSplit
    # # model.summary()
    es_callback = EarlyStopping(monitor='val_loss', patience=10)
    history = model.fit(X, y_train, 
                        epochs=nepochs, 
                        batch_size=batchSize, 
                        validation_split=validationSplit, 
                        validation_data=(X_tunn, y_Tunn),
                        # callbacks=[es_callback], 
                        verbose = 0)
    
    if savemode:
        Dtime = ''#datetime.now().strftime("%Y%m%d-%H%M%S")
        # saving whole model
        model.save(os.path.join(WEIGHTS_FOLDER, '1_ResNet50_Model_ep_'+str(nepochs)+'_val'+str(validationSplit).replace('.', '_')+'_batch'+str(batchSize)+'_'+Dtime+'.h5'))




if __name__ == '__main__':
    main()
        
