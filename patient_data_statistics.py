#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# ---------------------------------------------------------------------------
# Created on Tue Nov  2 08:23:07 2021
#
# @author: SeyedM.MousaviKahaki (mousavikahaki@gmail.com)
#----------------------------------------------------------------------------
# Title:        Patient data statistics
#
# Description:  This code reads the final dataset and plot different statistics
#               
#
#
# Input:        CSV file: Dataset 
# Output:       Plots: 
#
# 
# Example:      patient_data_statistics.py or runfile(patient_data_statistics.py)
#
#
# version ='3.0'
# ---------------------------------------------------------------------------
"""
Created on Mon Dec 20 09:24:29 2021

@author: SeyedM.MousaviKahaki
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
# from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
# from keras.models import Sequential
# from keras.layers import Dense
# from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, cohen_kappa_score


########### TO DO
# 
# Add columns:
#     Connect DatasetInfoFile to Patient file
#     Consider Exclude column
#     HAS CAH, HAS CA, HAS BENIGN area
#     CAH,CA, Benign AREA in pixels (3 columns)
#     Contour Features:
#         Check GECo slides
#   Assessment of different Classification/Transfer learning methods
#Check TAITANIC Dataset Analysis code(s)
###########


DATAFILE = 'C:/DATA/Aperio_dataset_v9.csv'


############### READ DATA
data = pd.read_csv(DATAFILE)

############### Describe Data
data.head()
data.columns
data.info()
desc = data.describe()
print(desc)


# ############### Select sub data (choose informative data)
# SVSSizes = data['SVS Rect']
# SVSResolution = data['SVS Resolution']

# BMIs = data['BMI']
# Races = data['Race']
# Truth = data['Responder']
# SVSMagnification = data['SVS Magnification']
# SVSMPP = data['SVS MPP']

# TotalCAHArea = data['Total CAH Area']                  
# TotalCarcinomaArea = data['Total Carcinoma Area']                 
# TotalBenignArea = data['Total BenignArea']
# CAH_Area_Micron = data['CAH_Area_Micron']
# Carcinoma_Area_Micron = data ['Carcinoma_Area_Micron']
# Benign_Area_Micron = data['Benign_Area_Micron']

# ##### Fix RACE
# data['Race'].value_counts()
data["Race"] = np.where(data["Race"] == 'WNH', "WHITE", data["Race"])
data["Race"] = np.where(data["Race"] == 'WH', "WHITE", data["Race"])
data["Race"] = np.where(data["Race"] == 'AANH', "AA", data["Race"])
data["Race"] = np.where(data["Race"] == 'AAH', "AA", data["Race"])
data["Race"] = np.where(data["Race"] == 'NAN', np.nan, data["Race"])
data['Race'].value_counts()

data.insert(8, 'CAH_Carcinoma_Area_Micron', np.nan)
data['CAH_Carcinoma_Area_Micron'] = data['Carcinoma_Area_Micron']+data['CAH_Area_Micron']
data.insert(8, 'ResponderNumeric', np.nan)
label_encoder = LabelEncoder()
data['ResponderNumeric'] = label_encoder.fit_transform(data['Responder'].astype(str))

##### Responder
data["Responder"] = np.where(data["Responder"] == 'Y', "Yes", data["Responder"])
data["Responder"] = np.where(data["Responder"] == 'N', "No", data["Responder"])
N_N = len(data[(data['Responder']== 'Yes')])
N_Y = len(data[(data['Responder']== 'No')])
percent = []
percent.append(np.round((N_N / (N_N+N_Y))* 100))
percent.append(np.round((N_Y / (N_N+N_Y))* 100))
ax = sns.countplot(x='Responder', data=data)
cnt = 0
for p in ax.patches:
   ax.annotate("N="+'{:.0f}'.format(p.get_height())+" ("+str(percent[cnt])+"%)", (p.get_x()+0.2, p.get_height()+0.06))
   cnt = cnt + 1
plt.title('Truth Distribution (N=91)')   
plt.ylabel('Count')
plt.show()


def plot_dist(data,col): 
    data=data.loc[data[col].notnull()]
    data=data.loc[data['ResponderNumeric'].notnull()]
    data[col] = data[col].astype(float).astype(int)
    data['ResponderNumeric'] = data['ResponderNumeric'].astype(float).astype(int)
    # sns.lmplot(data=data,x=col1,y=col2)
    g = sns.displot(data, x=col, hue='ResponderNumeric', kind="hist", fill=True)
    plt.legend(title='Responder', loc='upper right', labels=['Yes', 'No'])
    plt.show(g)
    g = sns.displot(data, x=col, hue='ResponderNumeric', kind="kde", fill=True)
    plt.legend(title='Responder', loc='upper right', labels=['Yes', 'No'])
    plt.show(g)

##### Age
data['ResponderNumeric'] = 1 - data['ResponderNumeric'] # Swith Colors
plot_dist(data,'Age at dx')
plot_dist(data,'BMI')
plot_dist(data,'CAH_Carcinoma_Area_Micron')
plot_dist(data,'Carcinoma_Area_Micron')
plot_dist(data,'CAH_Area_Micron')
plot_dist(data,'Race')

sns.pairplot(data[['Age at dx','BMI','CAH_Carcinoma_Area_Micron','Responder']], hue='Responder', height=3)

##################### Three VARIABLES - LINEAR MODEL PLOT
list(data)
sns.set_theme()

plt.figure(figsize=(20,6))
ax = sns.relplot(
    data=data,
    x="Age at dx", y="BMI", col="Responder",
    hue="Race", style="Race", size="CAH_Carcinoma_Area_Micron",
)
plt.figure(figsize=(20,6))
sns.relplot(
    data=data,
    x="Age at dx", y="BMI", col="Responder",
    hue="Race", style="Race", size="Carcinoma_Area_Micron",
)
plt.figure(figsize=(20,6))
sns.relplot(
    data=data,
    x="Age at dx", y="BMI", col="Responder",
    hue="Race", style="Race", size="CAH_Area_Micron",
)

##################### Size of WSI - Scatter
df = data['SVS Rect'].str.split(',', expand=True)
df = df.rename({0: 'Responder',1: 'SizeXY',2: 'X', 3: 'Y'}, axis=1)  # new method
df['Responder'] = data['Responder']
df['Y'] = df['Y'].str.replace(')','')
df['X']= df['X'].astype(np.int64)
df['Y'] = df['Y'].astype(np.int64)
df['SizeXY'] = df['X'] * df['Y']

df.insert(4, 'MPP', np.nan)
df.insert(5, 'Magnification', np.nan)
df.insert(6, 'Resolution', np.nan)
df.insert(7, 'File Size', np.nan)
df['MPP'] = data['SVS MPP']
df['Magnification'] = data['SVS Magnification']
df['Resolution'] = data['SVS Resolution']
df['File Size'] = data['SVS File Size']

g = sns.relplot(data=df, x='X', y='Y', kind='scatter',aspect=1.4,  
                palette='tab10', size='Magnification',hue='MPP')
g.set(ylim=(0, 100000))
plt.title('Whole Slide Image Information')
plt.xlabel('WSI Size (X)')
plt.ylabel('WSI Size (Y)')
plt.show(g)

# ############### Correlation
# label_encoder = LabelEncoder()
# data['Responder'] = label_encoder.fit_transform(data['Responder'].astype(str))
# data['Initial dx'] = label_encoder.fit_transform(data['Initial dx'].astype(str))
# data['Race'] = label_encoder.fit_transform(data['Race'].astype(str))
# data['DM (Y/N)'] = label_encoder.fit_transform(data['DM (Y/N)'].astype(str))
# data['Progestin Use (type/agent)'] = label_encoder.fit_transform(data['Progestin Use (type/agent)'].astype(str))
# data['FHx of endometrial CA'] = label_encoder.fit_transform(data['FHx of endometrial CA'].astype(str))
# data['PHx of breast/ovarian CA'] = label_encoder.fit_transform(data['PHx of breast/ovarian CA'].astype(str))
# print(data['Responder'].unique())

# corr = data.corr()
# matrix = np.triu(corr)
# sns.heatmap(data.corr(), annot=True,vmax=1,center=0,cmap='coolwarm',mask=matrix)

# sns.heatmap(corr,
#             xticklabels=corr.columns.values,
#             yticklabels=corr.columns.values)



# corr = data.corr(method='spearman')
# sns.heatmap(corr,
#             xticklabels=corr.columns.values,
#             yticklabels=corr.columns.values)



