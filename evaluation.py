#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
#
# ---------------------------------------------------------------------------
# Created on Fri Feb 11 10:17:47 2022
#
# @author: SeyedM.MousaviKahaki (mousavikahaki@gmail.com)
#----------------------------------------------------------------------------
# Title:        evaluation 
#
# Description:  This code import the embedded features, load model, and evaluate the model
#               
# Input:        String: Source directory where the WSIs and corresponding XML files are located  
# Output:       Extracted annotations
#
# 
# Example:      evaluation.py --testing_data TESTING_DATA --testing_labels TESTING_LABELS --weight_dir WEIGHT_DIR --model_file MODEL_FILE --random_forest RANDOM_FOREST --bootsrap BOOTSRAP
#               OR
#               For AE
#               runfile('evaluation.py', args='--testing_data "C:/DATA/2_extracted_cutted_Augmented/Features/AE/Testing/" --testing_labels "C:/DATA/2_extracted_cutted_Augmented/Features/AE/TestingLabels.h5" --weight_dir "models/saved_models/" --model_file "1_AE_Model_ep_5000_val0_1_batch32_.h5" --random_forest False --bootsrap False')
#               For ResNet50
#               runfile('evaluation.py', args='--testing_data "C:/DATA/2_extracted_cutted_Augmented/Features/ResNet50/Testing/" --testing_labels "C:/DATA/2_extracted_cutted_Augmented/Features/ResNet50/TestingLabels.h5" --weight_dir "models/saved_models/" --model_file "1_ResNet50_Model_ep_5000_val0_1_batch16_.h5" --random_forest False --bootsrap False')
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


from config import parameters

print(parameters.OUTPUTDIR_WIN)


# WEIGHTS_FOLDER = 'C:/DATA/Code/weights/'
# DATA_FOLDER = 'C:/DATA/extracted_cutted/data/AE_Data/'
# LOG_DIR = 'C:/DATA/Code/weights/logDir/'



# fo = open("C:/Users/SeyedM.MousaviKahaki/OneDrive - FDA/Documents/OWH_Project/WSI_Analysis/WSI/parameters.dat", "wb")
# fo.close()

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

parser.add_argument('--weight_dir', type = str,
					help='path to folder to load and save model weights')

parser.add_argument('--model_file', type = str,
					help='path to folder to load and save model weights')

parser.add_argument('--training_data', type = str,
					help='path to folder containing raw wsi image files')
parser.add_argument('--training_labels', type = str,
					help='path to folder to save extracted annotations')

parser.add_argument('--testing_data', type = str,
					help='path to folder containing raw wsi image files')
parser.add_argument('--testing_labels', type = str,
					help='path to folder to save extracted annotations')



add_boolean_argument(parser, 'random_forest', default=False)
add_boolean_argument(parser, 'bootsrap', default=False)


def load(self, filename="config/param_evl.dat"):
    d = {"Fn": "Fn", 
         "n_bootstraps": "n_bootstraps",
         "k": "k",
         "rng_seed": "rng_seed"
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
P = A()
load(P)
P.__dict__


def embedding_AE():
    
    args = parser.parse_args()
    INPUT_DIR = args.source
    WEIGHTS_FOLDER = args.weight_folder
    OUTPUT_DIR = args.save_dir
    # embedding_model = args.embedding_model
    AE_model_name = P.AE_model#'epc200_im256_batch256_20220222-104322_EncoderModel.h5'
    DatasetFile = P.DatasetFile#'C:/DATA/Aperio_dataset_v10.csv'
    filename_field = P.fname_field.replace('_' ,' ') # 'Filename of initial Aperio slide'
    truth_field = P.truth_field # 'Responder'
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
    
    with h5py.File(OUTPUT_DIR+'/TunningLabels.h5','w') as h5f:
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

def embedding_ResNet50():
    ############################################# Feature Extraction Using RESNET50 (Transfer Learning)
    args = parser.parse_args()
    INPUT_DIR = args.source
    # WEIGHTS_FOLDER = args.weight_folder
    OUTPUT_DIR = args.save_dir
    # embedding_model = args.embedding_model
    # AE_model_name = P.AE_model#'epc200_im256_batch256_20220222-104322_EncoderModel.h5'
    DatasetFile = P.DatasetFile#'C:/DATA/Aperio_dataset_v10.csv'
    filename_field = P.fname_field.replace('_' ,' ') # 'Filename of initial Aperio slide'
    truth_field = P.truth_field # 'Responder'
    
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


def main():
    
   
    args = parser.parse_args()

    # WEIGHTS_FOLDER = args.weightes_folder # 'C:/DATA/Code/weights/'
    
    testing_data = args.testing_data # C:/DATA/2_extracted_cutted_Augmented/Features/AE/Testing/
    testing_labels = args.testing_labels # C:/DATA/2_extracted_cutted_Augmented/Features/AE/TestingLabels.h5
    
    random_forest = args.random_forest
    bootsrap = args.bootsrap
    
    weight_dir = args.weight_dir
    model_file = args.model_file
    
    
    
    # testing_data = 'C:/DATA/2_extracted_cutted_Augmented/Features/ResNet50/Testing/'
    # testing_labels = 'C:/DATA/2_extracted_cutted_Augmented/Features/ResNet50/TestingLabels.h5'
    # random_forest = False
    # bootsrap = False
    
    # model_file = 'C:/DATA/Code/DigiPath_OWH/models/saved_models/1_ResNet50_Model_ep_5000_val0_1_batch16_.h5'
    #############################################  READ FEATURES From HDF5 Files
    

   
    ######## Load Testing Data
    dirs = os.listdir(testing_data)
    allDataTest = []
    # labelTest = []
    y_test = h5py.File(testing_labels)['Labels'][:]
    
    for dr in dirs:
        # print("Processing: "+dr)
        print(dr)
        # Read Features as HDF5   
        encodings_loaded = h5py.File(testing_data+dr)['Features'][:]    
        allDataTest.append(encodings_loaded)
    
    
    
    
    ################# Create Training and Test Dataframes from Data
    
    X_testO = []
    X_testO = pd.DataFrame(X_testO)
    for i in range(len(allDataTest)):
        print(i)
        print(len(allDataTest[i]))
        Adf=[]
        Adf = pd.DataFrame(allDataTest[i]).values.flatten()
        X_testO = pd.concat([X_testO, pd.DataFrame(Adf).T], axis=0)     
    X_test = X_testO.iloc[:, 0:int(P.Fn)]
    X_test.shape
    # ## Remove Nan Columns
    X_test = X_test.fillna(-1)
    X_test = X_test.dropna(axis=1, how='any')
    
    if model_file.find('AE') > 0:
        # selector = SelectKBest(f_regression, k=int(a.k))
        # X_test= selector.fit_transform(X_test, y_test)
        selector = pickle.load(open(weight_dir+"selectorAE.sav",'rb'))
        X_test= selector.transform(X_test)   
    else:
        selector = pickle.load(open(weight_dir+"selector.sav",'rb'))
        X_test= selector.transform(X_test)    
    X_test.shape
    
    
    
    
    ################## Load Trained Model
    print(model_file)
    model = load_model(weight_dir+model_file)
    # history_dict = json.load(open(os.path.join(WEIGHTS_FOLDER,'ALLDATA_Feac_epc5000_val0_1_batch32__History'), 'r'))
     
    #######                Predict Values
    
    y_pred = model.predict(X_test)
    # to compare predict and test
    y_pred[:15]
    y_test[:15]
    ###############                            Evaluate Model
    # evaluate the model
    _, test_acc = model.evaluate(X_test, y_test, verbose=0)
    print('Accuracy: Testing: %.3f' % (test_acc))
    ###############
    # score = model.evaluate(X_test, y_test,verbose=0)
    # print(score)
    # ###############  Confusion matrix
    # conf = confusion_matrix(y_test.round(), y_pred.round())
    # print("Confusion Matrix: ", conf)
    
    cmtx = pd.DataFrame(
        confusion_matrix(y_test.round(), y_pred.round(), labels=[1, 0]), 
        index=['true:yes', 'true:no'], 
        columns=['pred:yes', 'pred:no']
    )
    print(cmtx)
    
    ###############  Precision
    precision = precision_score(y_test.round(), y_pred.round()) #  average=Nonefor precision from each class
    print("Precision: ",round(precision,2))
    ############### Recall
    recall = recall_score(y_test.round(), y_pred.round())
    print("Recall: ",round(recall,2))
    # ############### F1 score
    from sklearn import metrics
    f1_score1 = metrics.f1_score(y_test.round(), y_pred.round())
    print("F1 Score: ",round(f1_score1,2))
    # ############### Cohen's kappa
    # cohen_kappa_score = metrics.cohen_kappa_score(y_test.round(), y_pred.round())
    # print("Cohen_Kappa Score: ",cohen_kappa_score)
    precision, recall, thresholds = precision_recall_curve(y_test.round(), y_pred.round())
    # calculate precision-recall AUC
    # print("Area Under the Curve (AUC): ",auc(recall, precision))
    # print(confusion_matrix(y_test.round(), y_pred.round()))
    # print(classification_report(y_test.round(), y_pred.round()))
    print('Accuracy: ' ,round(test_acc,2))
    
    ########################################## ROC CURVE (Kera vs Random Forest)
    y_pred_keras = model.predict(X_test).ravel()
    fpr_keras, tpr_keras, thresholds_keras = roc_curve(y_test, y_pred_keras,drop_intermediate=False)

    auc_keras = auc(fpr_keras, tpr_keras)
    print("AUC: ",round(auc_keras,2))
    
    plt.figure(1)
    plt.plot([0, 1], [0, 1], 'k--')
    plt.plot(fpr_keras, tpr_keras, label='Keras (area = {:.3f})'.format(auc_keras))
    # plt.plot(fpr_rf, tpr_rf, label='RF (area = {:.3f})'.format(auc_rf))
    plt.xlabel('False positive rate')
    plt.ylabel('True positive rate')
    plt.title('ROC curve i = ' + str(i))
    plt.legend(loc='best')
    plt.show()
    
    ## Save AUC Curve Data
    df = pd.DataFrame({'fpr':fpr_keras,'tpr':tpr_keras})  
    df.plot('fpr','tpr')   

    ##################### Bootsrap 
    if bootsrap:
        print("Original ROC area: {:0.2f}".format(roc_auc_score(y_test, y_pred_keras)))
        
        n_bootstraps = int(P.n_bootstraps)
        rng_seed = int(P.rng_seed)  # control reproducibility
        bootstrapped_scores = []
        
        rng = np.random.RandomState(rng_seed)
        for i in range(n_bootstraps):
            # bootstrap by sampling with replacement on the prediction indices
            indices = rng.randint(0, len(y_pred), len(y_pred))
            if len(np.unique(y_test[indices])) < 2:
                # We need at least one positive and one negative sample for ROC AUC
                # to be defined: reject the sample
                continue
        
            score = roc_auc_score(y_test[indices], y_pred[indices])
            bootstrapped_scores.append(score)
            print("Bootstrap #{} ROC area: {:0.2f}".format(i + 1, score))
            
        
        plt.hist(bootstrapped_scores, bins=50)
        plt.title('Histogram of the bootstrapped ROC AUC scores')
        plt.show()
    
    
    
    ######################### Renadom Forest
    
    if random_forest:
        # # Supervised transformation based on random forests
        # rf = RandomForestClassifier(max_depth=3, n_estimators=10)
        # rf.fit(X, y_train)
        
        # load, no need to initialize the loaded_rf
        loaded_rf = joblib.load("models/saved_models/random_forest.joblib3")
        # loaded_rf = rf
        
        y_pred_rf = loaded_rf.predict_proba(X_test)[:, 1]
        fpr_rf, tpr_rf, thresholds_rf = roc_curve(y_test, y_pred_rf)
        auc_rf = auc(fpr_rf, tpr_rf)
        
        
        plt.figure(1)
        plt.plot([0, 1], [0, 1], 'k--')
        plt.plot(fpr_keras, tpr_keras, label='Keras (area = {:.3f})'.format(auc_keras))
        plt.plot(fpr_rf, tpr_rf, label='RF (area = {:.3f})'.format(auc_rf))
        plt.xlabel('False positive rate')
        plt.ylabel('True positive rate')
        plt.title('ROC curve i = ' + str(i))
        plt.legend(loc='best')
        plt.show()
        # # Zoom in view of the upper left corner.
        # plt.figure(2)
        # plt.xlim(0, 0.2)
        # plt.ylim(0.8, 1)
        # plt.plot([0, 1], [0, 1], 'k--')
        # plt.plot(fpr_keras, tpr_keras, label='Keras (area = {:.3f})'.format(auc_keras))
        # plt.plot(fpr_rf, tpr_rf, label='RF (area = {:.3f})'.format(auc_rf))
        # plt.xlabel('False positive rate')
        # plt.ylabel('True positive rate')
        # plt.title('ROC curve (zoomed in at top left)')
        # plt.legend(loc='best')
        # plt.show()
        
        ###################################### Random Forest Evaluation
        cmtx = pd.DataFrame(
            confusion_matrix(y_test.round(), y_pred_rf.round(), labels=[1, 0]), 
            index=['true:yes', 'true:no'], 
            columns=['pred:yes', 'pred:no']
        )
        print(cmtx)
        ###############  Precision
        precision = precision_score(y_test.round(), y_pred_rf.round()) #  average=Nonefor precision from each class
        print("Precision: ",round(precision,2))
        ############### Recall
        recall = recall_score(y_test.round(), y_pred_rf.round())
        print("Recall: ",round(recall,2))
        # ############### F1 score
        from sklearn import metrics
        f1_score1 = metrics.f1_score(y_test.round(), y_pred_rf.round())
        print("F1 Score: ",round(f1_score1,2))
        # ############### Cohen's kappa
        # cohen_kappa_score = metrics.cohen_kappa_score(y_test.round(), y_pred_rf.round())
        # print("Cohen_Kappa Score: ",cohen_kappa_score)
        precision, recall, thresholds = precision_recall_curve(y_test.round(), y_pred_rf.round())
        # calculate precision-recall AUC
        # print("Area Under the Curve (AUC): ",auc(recall, precision))
        # print(confusion_matrix(y_test.round(), y_pred_rf.round()))
        # print(classification_report(y_test.round(), y_pred_rf.round()))
    
    
    

def save_data(model,X,X_test,y_train,y_test,nepochs,validationSplit,batchSize,history,WEIGHTS_FOLDER):

    NN= 'Final1/'
    Dtime = ''#datetime.now().strftime("%Y%m%d-%H%M%S")
    # saving whole model
    model.save(os.path.join(WEIGHTS_FOLDER, 'CL/All_Data_OKL_APERIO/'+NN+'ALLDATA_Feac_epc'+str(nepochs)+'_val'+str(validationSplit).replace('.', '_')+'_batch'+str(batchSize)+'_'+Dtime+'_Model.h5'))
    # Saving model history
    np.save(os.path.join(WEIGHTS_FOLDER, 'CL/All_Data_OKL_APERIO/'+NN+'ALLDATA_Feac_epc'+str(nepochs)+'_val'+str(validationSplit).replace('.', '_')+'_batch'+str(batchSize)+'_'+Dtime+'_History.npy'),history.history)
    
    # Save train and test Data
    np.save(os.path.join(WEIGHTS_FOLDER, 'CL/All_Data_OKL_APERIO/'+NN+'ALLDATA_X_trainKBest200'),X)
    np.save(os.path.join(WEIGHTS_FOLDER, 'CL/All_Data_OKL_APERIO/'+NN+'ALLDATA_X_testKBest200'),X_test)
    np.save(os.path.join(WEIGHTS_FOLDER, 'CL/All_Data_OKL_APERIO/'+NN+'ALLDATA_y_trainKBest'),y_train)
    np.save(os.path.join(WEIGHTS_FOLDER, 'CL/All_Data_OKL_APERIO/'+NN+'ALLDATA_y_testKBest'),y_test)
    
    # Get the dictionary containing each metric and the loss for each epoch
    history_dict = history.history
    # Save it under the form of a json file
    json.dump(history_dict, open(os.path.join(WEIGHTS_FOLDER, 'CL/All_Data_OKL_APERIO/'+NN+'ALLDATA_Feac_epc'+str(nepochs)+'_val'+str(validationSplit).replace('.', '_')+'_batch'+str(batchSize)+'_'+Dtime+'_History'), 'w'))
    
    
    
    
if __name__ == '__main__':
    main()
        