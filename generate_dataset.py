#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# ---------------------------------------------------------------------------
# Created on Thu Oct 14 08:35:13 2021
#
# @author: SeyedM.MousaviKahaki (mousavikahaki@gmail.com)
#----------------------------------------------------------------------------
# Title:        Generate Dataset
#
# Description:  This code Generate the main dataset by combining the patient 
#               Excel file and the information from the Whole Slide Images
#
#
# Input:        CSV and SVS files: Patient data and WSIs 
# Output:       CSV file: Final Dataset
#
# 
# Example:      generate_dataset.py or runfile(generate_dataset.py)
#
#
# version ='3.0'
# ---------------------------------------------------------------------------
 
##############################   General Imports
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
import seaborn as sns
import sys
##############################   Internal Imports
sys.path.insert(1, '../Config/')
import parameters

############### Input
DATAFILE_patient_list = parameters.DATAFILE_patient_list
DATAFILE_dataset_features = parameters.DATAFILE_dataset_features
OUTPUTDIR_WIN = parameters.OUTPUTDIR_WIN 
OUTPUTDIR_UBUNTU = parameters.OUTPUTDIR_UBUNTU
DATASETNAME = parameters.DATASETNAME
ver = 'v10'
############### READ DATA
patient_list = pd.read_csv('C:/DATA/Master patient list for FDA 9-20-21_Seyed_v3.csv')#DATAFILE_patient_list
dataset_features = pd.read_csv('C:/DATA/Aperio_datasetInfo_v2.csv')
# dataset_features = pd.read_csv('C:/DATA/3Dhistech_datasetInfo.csv') #_Histech
dataset_features_Okl = pd.read_csv('C:/DATA/Aperio_Oklahoma_datasetInfo_v1.csv')
# dataset_features_Okl = pd.read_csv('C:/DATA/3DHistech_Oklahoma_datasetInfo_v1.csv') #_H istech
dataset_features_Okl = dataset_features_Okl.drop('Responder', 1)
############### Describe Data
patient_list.head()
patient_list.columns
patient_list.info()
desc = patient_list.describe()
print(desc)
dataset_features.head()
dataset_features.columns
dataset_features.info()
desc = dataset_features.describe()
print(desc)

############### Select sub data (choose informative data)

# remove Data without Aperio image
patient_list['Exclude'].unique()
patient_list = patient_list[(patient_list['Exclude']==0)]
# remove reason for exclusion field
patient_list = patient_list.drop(['Exclude',
                                  'Reason for Exclusion'
                                  ],axis=1)

# Fix Truth
patient_list = patient_list.rename({'Responder?': 'Responder'}, axis=1) 
############### Uppercase all columns and proper nan
patient_list = patient_list.apply(lambda x: x.astype(str).str.upper())
dataset_features = dataset_features.apply(lambda x: x.astype(str).str.upper())
dataset_features_Okl = dataset_features_Okl.apply(lambda x: x.astype(str).str.upper())

patient_list = patient_list.replace('NAN',np.nan)
dataset_features = dataset_features.replace('NAN',np.nan)
dataset_features_Okl = dataset_features_Okl.replace('NAN',np.nan)


############### Fix BMI
#Use BMI at dx if not use at flollowup
# add BMI field after BMI at follow-up field
patient_list.insert(9, 'BMI', np.nan)
patient_list['BMI'] = patient_list['BMI at dx (kg)']
patient_list.BMI.fillna(patient_list['BMI at follow-up (kg)'], inplace=True)
del patient_list['BMI at dx (kg)']
del patient_list['BMI at follow-up (kg)']

patient_list.insert(8, 'BMICAT', np.nan)
# Numbers calculated with the code at the end of this file
patient_list['BMICAT'] = np.where(pd.to_numeric(patient_list['BMI']) < 35.9, 'L_BMI',patient_list['BMICAT'])
patient_list['BMICAT'] = np.where((pd.to_numeric(patient_list['BMI']) > 35.9) & (pd.to_numeric(patient_list['BMI']) < 43.885), 'M_BMI',patient_list['BMICAT'])
patient_list['BMICAT'] = np.where((pd.to_numeric(patient_list['BMI']) > 43.885) & (pd.to_numeric(patient_list['BMI']) < 53.8125), 'H_BMI',patient_list['BMICAT'])
patient_list['BMICAT'] = np.where(pd.to_numeric(patient_list['BMI']) > 53.8125, 'VH_BMI',patient_list['BMICAT'])


############### Fix RACE
patient_list['Race'].value_counts()
patient_list["Race"] = np.where(patient_list["Race"] == 'W', "WHITE", patient_list["Race"])
patient_list['Race'].value_counts()

############### Merge DataFrames
AllData0 = pd.merge(patient_list,dataset_features,on="Patient ID", right_index=False) # can add how='inner', left_on='left column',right_on='right column'
AllData1 = pd.merge(patient_list,dataset_features_Okl,on="Patient ID", right_index=False) # can add how='inner', left_on='left column',right_on='right column'
AllData = pd.concat([AllData0,AllData1], axis=0)

############### Save Dataset
if os.name == 'nt':
    print("File Saved on Windows!")
    AllData.to_csv(OUTPUTDIR_WIN +DATASETNAME+'_dataset_'+ver+'.csv',index=False)
else:
    print("File Saved on Ubuntu!")  
    AllData.to_csv(OUTPUTDIR_UBUNTU +DATASETNAME+'_dataset_'+ver+'.csv',index=False)


### Plot Responders Count
data = AllData
##### Responder
data["Responder"] = np.where(data["Responder"] == 'Y', "Yes", data["Responder"])
data["Responder"] = np.where(data["Responder"] == 'N', "No", data["Responder"])
N_N = len(data[(data['Responder']== 'Yes')])
N_Y = len(data[(data['Responder']== 'No')])
total = N_N + N_Y
percent = []
percent.append(np.round((N_N / (N_N+N_Y))* 100))
percent.append(np.round((N_Y / (N_N+N_Y))* 100))
ax = sns.countplot(x='Responder', data=data)
cnt = 0
for p in ax.patches:
   ax.annotate("N="+'{:.0f}'.format(p.get_height())+" ("+str(percent[cnt])+"%)", (p.get_x()+0.2, p.get_height()+0.06))
   cnt = cnt + 1
plt.title('Truth Distribution (N='+str(total)+')')   
plt.ylabel('Count')
plt.show()