#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
#
# ---------------------------------------------------------------------------
# Created on Tue Nov  2 08:23:07 2021
#
# @author: SeyedM.MousaviKahaki (mousavikahaki@gmail.com)
#----------------------------------------------------------------------------
# Title:        Train Test Split
#
# Description:  This code Split the data into training and test subsets
#               by stratification over several variables (columns)
#
#
# Input:        CSV file: Dataset 
# Output:       CSV file: training and test subsets
#
# 
# Example:      train_test_split.py --outputdir OUTPUT_DIRECTORY --datasetname DATASET_NAME --datasetfile DATASET_FILE --datasetversion DATASET_VERSION
#               OR
#               runfile('train_test_split.py', args='--outputdir "C:/DATA/" --datasetname "Aperio" --datasetfile "C:/DATA/Aperio_dataset_v10.csv" --datasetversion "v11"')
#
#
#
# version ='3.0'
# ---------------------------------------------------------------------------
"""
##############################   General Imports
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.model_selection import train_test_split
import sys
import argparse


parser = argparse.ArgumentParser(description='train test split')


parser.add_argument('--outputdir', type = str,
					help='path to the ourput directory')
parser.add_argument('--datasetname', type = str,
					help='name of the dataset')
parser.add_argument('--datasetfile', type = str,
					help='path to the dataset file')
parser.add_argument('--datasetversion', type = str,
					help='the dataset version')


def main():
    
    args = parser.parse_args()
    
    OUTPUTDIR_WIN = args.outputdir
    DATASETNAME = args.datasetname
    DATAFILE = args.datasetfile
    ver = args.datasetversion
    ############### READ DATA
    data = pd.read_csv(DATAFILE)
    
    ############### Lowecase all columns -------------------------------------------
    data = data.apply(lambda x: x.astype(str).str.upper())
    
    ############### Describe Data
    data.head()
    data.columns
    data.info()
    desc = data.describe()
    print(desc)
    list(data)
    
    # Extra Info
    # data.iloc[: , 5:10].describe()
    
    # Plot Count with Percentage
    L_BMI = len(data[(data['BMICAT']== 'L_BMI')])
    M_BMI = len(data[(data['BMICAT']== 'M_BMI')])
    H_BMI = len(data[(data['BMICAT']== 'H_BMI')])
    VH_BMI = len(data[(data['BMICAT']== 'VH_BMI')])
    percent = []
    percent.append(np.round((L_BMI / (L_BMI+M_BMI+H_BMI+VH_BMI))* 100))
    percent.append(np.round((M_BMI / (M_BMI+M_BMI+H_BMI+VH_BMI))* 100))
    percent.append(np.round((H_BMI / (H_BMI+M_BMI+H_BMI+VH_BMI))* 100))
    percent.append(np.round((VH_BMI / (VH_BMI+M_BMI+H_BMI+VH_BMI))* 100))
    ax = sns.countplot(x='BMICAT', data=data,order=[ 'L_BMI','M_BMI', 'H_BMI', 'VH_BMI'])
    cnt = 0
    for p in ax.patches:
       ax.annotate("cnt:"+'{:.0f}'.format(p.get_height())+"  %"+str(percent[cnt]), (p.get_x()+0.1, p.get_height()+0.06))
       cnt = cnt + 1
    plt.show()
    
    
    ############### Select sub data (choose informative data)
    
    data = data[['Patient ID',
     'Initial dx',
     'Filename of initial Aperio slide',
     'Filename of initial 3D Histech slide',
     'Responder',
     'Age at dx',
     'Race',
     'DM (Y/N)',
     'Progestin Use (type/agent)',
     'FHx of endometrial CA',
     'PHx of breast/ovarian CA',
     'BMI',
     'BMICAT',
     'SVS File Name',
     'SVS File Size',
     'SVS Magnification',
     'SVS MPP',
     'SVS MetaData']]
    
    
    
    
    
    ############### Describe sub data
    data.columns
    data.info()
    desc = data.describe()
    print(desc)
    desc = data.astype('object').describe().transpose()
    print(desc)
    
    # Unique values
    data['Responder'].unique()
    data['Race'].unique()
    data['BMICAT'].unique()
    data['SVS Magnification'].unique()
   
    
    
    
    ############### Class Distributions
    ##### Responder
    N_N = len(data[(data['Responder']== 'Y')])
    N_Y = len(data[(data['Responder']== 'N')])
    percent = []
    percent.append(np.round((N_N / (N_N+N_Y))* 100))
    percent.append(np.round((N_Y / (N_N+N_Y))* 100))
    ax = sns.countplot(x='Responder', data=data)
    cnt = 0
    for p in ax.patches:
       ax.annotate("cnt:"+'{:.0f}'.format(p.get_height())+"  percent:"+str(percent[cnt])+"%", (p.get_x()+0.1, p.get_height()+0.06))
       cnt = cnt + 1
    plt.show()
    # ##### Fix RACE
    # data['Race'].value_counts()
    data["Race"] = np.where(data["Race"] == 'WNH', "WHITE", data["Race"])
    data["Race"] = np.where(data["Race"] == 'WH', "WHITE", data["Race"])
    data["Race"] = np.where(data["Race"] == 'AANH', "AA", data["Race"])
    data["Race"] = np.where(data["Race"] == 'AAH', "AA", data["Race"])
    data["Race"] = np.where(data["Race"] == 'NAN', np.nan, data["Race"])
    data['Race'].value_counts()
    
    ##### Magnification
    N_N = len(data[(data['SVS Magnification']== '40.0')])
    N_Y = len(data[(data['SVS Magnification']== '20.0')])
    percent = []
    percent.append(np.round((N_N / (N_N+N_Y))* 100))
    percent.append(np.round((N_Y / (N_N+N_Y))* 100))
    ax = sns.countplot(x='SVS Magnification', data=data)
    cnt = 0
    for p in ax.patches:
       ax.annotate("cnt:"+'{:.0f}'.format(p.get_height())+"  percent:"+str(percent[cnt])+"%", (p.get_x()+0.1, p.get_height()+0.06))
       cnt = cnt + 1
    plt.show()
    
    
    # ##### Fix RACE
    # data['Race'].value_counts()
    data["Race"] = np.where(data["Race"] == 'WNH', "WHITE", data["Race"])
    data["Race"] = np.where(data["Race"] == 'WH', "WHITE", data["Race"])
    data["Race"] = np.where(data["Race"] == 'AANH', "AA", data["Race"])
    data["Race"] = np.where(data["Race"] == 'AAH', "AA", data["Race"])
    data["Race"] = np.where(data["Race"] == 'NAN', np.nan, data["Race"])
    data['Race'].value_counts()
    
    ##### RACE
    N_WHITE = len(data[(data['Race']== 'WHITE')])
    N_AA = len(data[(data['Race']== 'AA')])
    N_ASIAN = len(data[(data['Race']== 'ASIAN')])
    percent = []
    percent.append(np.round((N_WHITE / (N_WHITE+N_AA+N_ASIAN))* 100))
    percent.append(np.round((N_AA / (N_WHITE+N_AA+N_ASIAN))* 100))
    percent.append(np.round((N_ASIAN / (N_WHITE+N_AA+N_ASIAN))* 100))
    ax = sns.countplot(x='Race', data=data,order=['WHITE','AA','ASIAN'])
    cnt = 0
    for p in ax.patches:
        ax.annotate("cnt:"+'{:.0f}'.format(p.get_height())+"  percent:"+str(percent[cnt])+"%", (p.get_x()-0.05, p.get_height()+0.06))
        cnt = cnt + 1
    plt.show()
    
    ############### Train Test Split
    # data1 = data.dropna(subset=['Race'])
    data1 = data
    list(data)
    
    data1 = data1.rename(columns={'Responder?': 'Responder'})
    X = data1[['Patient ID',
     'Initial dx',
     'Filename of initial Aperio slide',
     'Filename of initial 3D Histech slide',
     'Responder',
     'Age at dx',
     # 'Race',
     'BMI',
     'BMICAT',
     'SVS File Name',
     'SVS File Size',
     'SVS Magnification',
     'SVS MPP',
     'SVS MetaData']]
    X = X.dropna()
    
    
    y = data1[['Responder']]
    
    X['Responder'].value_counts()
    # X['Race'].value_counts()
    X['BMICAT'].value_counts()
    X['SVS Magnification'].value_counts()
    
    
    
        
    X_train1, X_test1 = train_test_split(X , test_size=0.2, random_state=20, stratify=X[['Responder','SVS Magnification']])
    

    ## Count Nulls
    X_test2 = X_test1.astype(object).replace('NAN', np.nan)
    numNulls = X_test2['Filename of initial 3D Histech slide'].isnull().values.sum()
    print(numNulls)
    ############### Save Dataset
    
    print("File Saved on Windows!")
    X_train1.to_csv(OUTPUTDIR_WIN +DATASETNAME+'_TrainingSet_'+ver+'.csv',index=False)
    X_test1.to_csv(OUTPUTDIR_WIN +DATASETNAME+'_TestSet_'+ver+'.csv',index=False)
    
    
    ##### Magnification Original
    N_N = len(data[(data['SVS Magnification']== '40.0')])
    N_Y = len(data[(data['SVS Magnification']== '20.0')])
    percent = []
    percent.append(np.round((N_N / (N_N+N_Y))* 100))
    percent.append(np.round((N_Y / (N_N+N_Y))* 100))
    ax = sns.countplot(x='SVS Magnification', data=data)
    cnt = 0
    for p in ax.patches:
       ax.annotate("cnt:"+'{:.0f}'.format(p.get_height())+"  percent:"+str(percent[cnt])+"%", (p.get_x()+0.1, p.get_height()+0.06))
       cnt = cnt + 1
    plt.title("Magnification - Dataset")
    plt.show()
    ##### Magnification Train
    N_N = len(X_train1[(X_train1['SVS Magnification']== '40.0')])
    N_Y = len(X_train1[(X_train1['SVS Magnification']== '20.0')])
    percent = []
    percent.append(np.round((N_N / (N_N+N_Y))* 100))
    percent.append(np.round((N_Y / (N_N+N_Y))* 100))
    ax = sns.countplot(x='SVS Magnification', data=X_train1)
    cnt = 0
    for p in ax.patches:
       ax.annotate("cnt:"+'{:.0f}'.format(p.get_height())+"  percent:"+str(percent[cnt])+"%", (p.get_x()+0.1, p.get_height()+0.06))
       cnt = cnt + 1
    plt.title("Magnification - Training Set")
    plt.show()
    ##### Magnification Train
    N_N = len(X_train1[(X_train1['SVS Magnification']== '40.0')])
    N_Y = len(X_train1[(X_train1['SVS Magnification']== '20.0')])
    percent = []
    percent.append(np.round((N_N / (N_N+N_Y))* 100))
    percent.append(np.round((N_Y / (N_N+N_Y))* 100))
    ax = sns.countplot(x='SVS Magnification', data=X_train1, order=['40.0','20.0'])
    cnt = 0
    for p in ax.patches:
       ax.annotate("cnt:"+'{:.0f}'.format(p.get_height())+"  percent:"+str(percent[cnt])+"%", (p.get_x()+0.1, p.get_height()+0.06))
       cnt = cnt + 1
    plt.title("Magnification - Train Set")
    plt.show()
    ##### Magnification Test
    N_N = len(X_test1[(X_test1['SVS Magnification']== '40.0')])
    N_Y = len(X_test1[(X_test1['SVS Magnification']== '20.0')])
    percent = []
    percent.append(np.round((N_N / (N_N+N_Y))* 100))
    percent.append(np.round((N_Y / (N_N+N_Y))* 100))
    ax = sns.countplot(x='SVS Magnification', data=X_test1, order=['40.0','20.0'])
    cnt = 0
    for p in ax.patches:
       ax.annotate("cnt:"+'{:.0f}'.format(p.get_height())+"  percent:"+str(percent[cnt])+"%", (p.get_x()+0.1, p.get_height()+0.06))
       cnt = cnt + 1
    plt.title("Magnification - Test Set")
    plt.show()
    
    
    
    
    ##### Responder Original
    N_N = len(data[(data['Responder']== 'Y')])
    N_Y = len(data[(data['Responder']== 'N')])
    percent = []
    percent.append(np.round((N_N / (N_N+N_Y))* 100))
    percent.append(np.round((N_Y / (N_N+N_Y))* 100))
    ax = sns.countplot(x='Responder', data=data)
    cnt = 0
    for p in ax.patches:
       ax.annotate("cnt:"+'{:.0f}'.format(p.get_height())+"  percent:"+str(percent[cnt])+"%", (p.get_x()+0.1, p.get_height()+0.06))
       cnt = cnt + 1
    plt.title("Truth - Dataset")
    plt.show()
    ##### Responder Train
    N_N = len(X_train1[(X_train1['Responder']== 'Y')])
    N_Y = len(X_train1[(X_train1['Responder']== 'N')])
    percent = []
    percent.append(np.round((N_N / (N_N+N_Y))* 100))
    percent.append(np.round((N_Y / (N_N+N_Y))* 100))
    ax = sns.countplot(x='Responder', data=X_train1,order=['Y','N'])
    cnt = 0
    for p in ax.patches:
       ax.annotate("cnt:"+'{:.0f}'.format(p.get_height())+"  percent:"+str(percent[cnt])+"%", (p.get_x()+0.1, p.get_height()+0.06))
       cnt = cnt + 1
    plt.title("Truth - Training Set")
    plt.show()
    ##### Responder Test
    N_N = len(X_test1[(X_test1['Responder']== 'Y')])
    N_Y = len(X_test1[(X_test1['Responder']== 'N')])
    percent = []
    percent.append(np.round((N_N / (N_N+N_Y))* 100))
    percent.append(np.round((N_Y / (N_N+N_Y))* 100))
    ax = sns.countplot(x='Responder', data=X_test1,order=['Y','N'])
    cnt = 0
    for p in ax.patches:
       ax.annotate("cnt:"+'{:.0f}'.format(p.get_height())+"  percent:"+str(percent[cnt])+"%", (p.get_x()+0.1, p.get_height()+0.06))
       cnt = cnt + 1
    plt.title("Truth - Test Set")
    plt.show()
    
    
    
    
    ##### BMICAT Original
    L_BMI = len(X[(X['BMICAT']== 'L_BMI')])
    M_BMI = len(X[(X['BMICAT']== 'M_BMI')])
    H_BMI = len(X[(X['BMICAT']== 'H_BMI')])
    VH_BMI = len(X[(X['BMICAT']== 'VH_BMI')])
    percent = []
    percent.append(np.round((L_BMI / (L_BMI+M_BMI+H_BMI+VH_BMI))* 100))
    percent.append(np.round((M_BMI / (M_BMI+M_BMI+H_BMI+VH_BMI))* 100))
    percent.append(np.round((H_BMI / (H_BMI+M_BMI+H_BMI+VH_BMI))* 100))
    percent.append(np.round((VH_BMI / (VH_BMI+M_BMI+H_BMI+VH_BMI))* 100))
    ax = sns.countplot(x='BMICAT', data=X,order=[ 'L_BMI','M_BMI', 'H_BMI', 'VH_BMI'])
    cnt = 0
    for p in ax.patches:
       ax.annotate("cnt:"+'{:.0f}'.format(p.get_height())+"("+str(percent[cnt])+"%)", (p.get_x()+0.1, p.get_height()+0.06))
       cnt = cnt + 1
    plt.title("BMI - Dataset")
    plt.show()
    ##### BMICAT Train
    L_BMI = len(X_train1[(X_train1['BMICAT']== 'L_BMI')])
    M_BMI = len(X_train1[(X_train1['BMICAT']== 'M_BMI')])
    H_BMI = len(X_train1[(X_train1['BMICAT']== 'H_BMI')])
    VH_BMI = len(X_train1[(X_train1['BMICAT']== 'VH_BMI')])
    percent = []
    percent.append(np.round((L_BMI / (L_BMI+M_BMI+H_BMI+VH_BMI))* 100))
    percent.append(np.round((M_BMI / (M_BMI+M_BMI+H_BMI+VH_BMI))* 100))
    percent.append(np.round((H_BMI / (H_BMI+M_BMI+H_BMI+VH_BMI))* 100))
    percent.append(np.round((VH_BMI / (VH_BMI+M_BMI+H_BMI+VH_BMI))* 100))
    ax = sns.countplot(x='BMICAT', data=X_train1,order=[ 'L_BMI','M_BMI', 'H_BMI', 'VH_BMI'])
    cnt = 0
    for p in ax.patches:
       ax.annotate("cnt:"+'{:.0f}'.format(p.get_height())+"("+str(percent[cnt])+"%)", (p.get_x()+0.1, p.get_height()+0.06))
       cnt = cnt + 1
    plt.title("BMI - Training Set")
    plt.show()
    ##### BMICAT Test
    L_BMI = len(X_test1[(X_test1['BMICAT']== 'L_BMI')])
    M_BMI = len(X_test1[(X_test1['BMICAT']== 'M_BMI')])
    H_BMI = len(X_test1[(X_test1['BMICAT']== 'H_BMI')])
    VH_BMI = len(X_test1[(X_test1['BMICAT']== 'VH_BMI')])
    percent = []
    percent.append(np.round((L_BMI / (L_BMI+M_BMI+H_BMI+VH_BMI))* 100))
    percent.append(np.round((M_BMI / (M_BMI+M_BMI+H_BMI+VH_BMI))* 100))
    percent.append(np.round((H_BMI / (H_BMI+M_BMI+H_BMI+VH_BMI))* 100))
    percent.append(np.round((VH_BMI / (VH_BMI+M_BMI+H_BMI+VH_BMI))* 100))
    ax = sns.countplot(x='BMICAT', data=X_test1,order=[ 'L_BMI','M_BMI', 'H_BMI', 'VH_BMI'])
    cnt = 0
    for p in ax.patches:
       ax.annotate("cnt:"+'{:.0f}'.format(p.get_height())+"  %"+str(percent[cnt]), (p.get_x()+0.1, p.get_height()+0.06))
       cnt = cnt + 1
    plt.title("BMI - Test Set")
    plt.show()
    
    
    return X_train1, X_test1



if __name__ == '__main__':
    main()

