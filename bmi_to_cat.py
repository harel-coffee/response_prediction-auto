#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# ---------------------------------------------------------------------------
# Created on Tue Nov  2 08:23:07 2021
#
# @author: SeyedM.MousaviKahaki (mousavikahaki@gmail.com)
#----------------------------------------------------------------------------
# Title:        Train Test Split
#
# Description:  This code converts BMI field to BMI category
#               
#
#
# Input:        CSV file: Patient List File
# Output:       CSV file: New Patient List File
#
# 
# Example:      BMI_Category.py or runfile(BMI_Category.py)
#
#
# version ='3.0'
# ---------------------------------------------------------------------------
"""
Created on Thu Oct 14 08:35:13 2021

@author: SeyedM.MousaviKahaki
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
import seaborn as sns


DATAFILE_patient_list = 'C:/DATA/Master patient list for FDA 9-20-21_Seyed_v2.csv'
DATAFILE_dataset_features = 'C:/DATA/Aperio_datasetInfo_v2.csv'
OUTPUTDIR_WIN = 'C:/DATA/'
OUTPUTDIR_UBUNTU = '/home/seyedm.mousavikahaki/Documents/'
DATASETNAME = 'Aperio'
ver = 'v5'
############### READ DATA
patient_list = pd.read_csv(DATAFILE_patient_list)
dataset_features = pd.read_csv(DATAFILE_dataset_features)

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


############### Uppercase all columns and proper nan
patient_list = patient_list.apply(lambda x: x.astype(str).str.upper())
dataset_features = dataset_features.apply(lambda x: x.astype(str).str.upper())

patient_list = patient_list.replace('NAN',np.nan)
dataset_features = dataset_features.replace('NAN',np.nan)


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
# Auto Generated
# Cutoff1 = 35.9
# Cutoff2 = 43.885
# Cutoff3 = 53.8125

# CDC
Cutoff1 = 18
Cutoff2 = 25
Cutoff3 = 30

# # CDC Obssey
# Cutoff1 = 30
# Cutoff2 = 35
# Cutoff3 = 40


patient_list['BMICAT'] = np.where(pd.to_numeric(patient_list['BMI']) < Cutoff1, 'L_BMI',patient_list['BMICAT'])
patient_list['BMICAT'] = np.where((pd.to_numeric(patient_list['BMI']) > Cutoff1) & (pd.to_numeric(patient_list['BMI']) < Cutoff2), 'M_BMI',patient_list['BMICAT'])
patient_list['BMICAT'] = np.where((pd.to_numeric(patient_list['BMI']) > Cutoff2) & (pd.to_numeric(patient_list['BMI']) < Cutoff3), 'H_BMI',patient_list['BMICAT'])
patient_list['BMICAT'] = np.where(pd.to_numeric(patient_list['BMI']) > Cutoff3, 'VH_BMI',patient_list['BMICAT'])




# Plot Count with Percentage
L_BMI = len(patient_list[(patient_list['BMICAT']== 'L_BMI')])
M_BMI = len(patient_list[(patient_list['BMICAT']== 'M_BMI')])
H_BMI = len(patient_list[(patient_list['BMICAT']== 'H_BMI')])
VH_BMI = len(patient_list[(patient_list['BMICAT']== 'VH_BMI')])
percent = []
percent.append(np.round((L_BMI / (L_BMI+M_BMI+H_BMI+VH_BMI))* 100))
percent.append(np.round((M_BMI / (M_BMI+M_BMI+H_BMI+VH_BMI))* 100))
percent.append(np.round((H_BMI / (H_BMI+M_BMI+H_BMI+VH_BMI))* 100))
percent.append(np.round((VH_BMI / (VH_BMI+M_BMI+H_BMI+VH_BMI))* 100))
ax = sns.countplot(x='BMICAT', data=patient_list,order=[ 'L_BMI','M_BMI', 'H_BMI', 'VH_BMI'])
cnt = 0
for p in ax.patches:
    ax.annotate("cnt:"+'{:.0f}'.format(p.get_height())+"  %"+str(percent[cnt]), (p.get_x()+0.1, p.get_height()+0.06))
    cnt = cnt + 1
plt.show()





