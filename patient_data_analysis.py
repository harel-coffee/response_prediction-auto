#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# ---------------------------------------------------------------------------
# Created on Wed Sep 29 10:35:19 2021
#
# @author: SeyedM.MousaviKahaki (mousavikahaki@gmail.com)
#----------------------------------------------------------------------------
# Title:        Patirnt Data Analysis
#
# Description:  This code Analysis the patient data and plot related information
#               
#
#
# Input:        CSV file: Original Patient Dataset File   
# Output:       None
#
# 
# Example:      train_test_split.py or runfile(train_test_split.py)
#
#
# version ='3.0'
# ---------------------------------------------------------------------------
"""
Created on Wed Sep 29 10:35:19 2021

@author: SeyedM.MousaviKahaki
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from keras.models import Sequential
from keras.layers import Dense
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, cohen_kappa_score


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


DATAFILE = 'C:/DATA/Master patient list for FDA 9-20-21_Seyed_v2.csv'


############### READ DATA
data = pd.read_csv(DATAFILE)

############### Describe Data
data.head()
data.columns
data.info()
desc = data.describe()
print(desc)


############### Select sub data (choose informative data)

# remove Data without Aperio image
data['Filename of initial Aperio slide'].unique()
data = data[(~data['Filename of initial Aperio slide'].isnull())]

datadesc = data[['Initial dx',
       'Responder?', 'Age at dx',
       'BMI at dx (kg)', 'BMI at follow-up (kg)', 'Race', 'DM (Y/N)',
       'Progestin Use (type/agent)', 'FHx of endometrial CA',
       'PHx of breast/ovarian CA']]


############### Lowecase all columns -------------------------------------------
datadesc = datadesc.apply(lambda x: x.astype(str).str.lower())

############### Remove nan responders
datadesc = datadesc[(datadesc['Responder?']== 'n') | (datadesc['Responder?']== 'y')]

############### Describe sub data
datadesc.columns
datadesc.info()
desc = datadesc.describe()
print(desc)
desc = datadesc.astype('object').describe().transpose()
print(desc)

# Unique values
datadesc['Initial dx'].unique()
datadesc['Responder?'].unique()
datadesc['Race'].unique()
datadesc['DM (Y/N)'].unique()
datadesc['Progestin Use (type/agent)'].unique()
datadesc['FHx of endometrial CA'].unique()
datadesc['PHx of breast/ovarian CA'].unique()

#### Class Distributions
N_N = len(datadesc[(datadesc['Responder?']== 'n')])
N_Y = len(datadesc[(datadesc['Responder?']== 'y')])
sns.countplot(x='Responder?', data=datadesc)

############### Find Missing Values
# sns.heatmap(datadesc=='nan',yticklabels=False,cbar=False,cmap='viridis')
datadesc = datadesc.replace('nan',np.nan)
sns.heatmap(datadesc.isnull(),yticklabels=False,cbar=False,cmap='viridis')
# Count Nulls
tempd = datadesc.isnull().sum()
# if nulls means anything
# datadesc.groupby(datadesc['Race'].isnull()).mean()

####### MatPlot Histogram
responders = datadesc[datadesc['Responder?']=='y']
N_responders = datadesc[datadesc['Responder?']=='n']
# col = 'Age at dx'
# col = 'BMI at dx (kg)'
col = 'BMI at follow-up (kg)'
responders=responders.loc[responders[col].notnull()]
N_responders=N_responders.loc[N_responders[col].notnull()]
maxx = 20
fig, ax = plt.subplots(1, 2)
ax[0].hist(responders[col].astype(float).astype(int),10,facecolor='green', alpha=0.5)
ax[1].hist(N_responders[col].astype(float).astype(int),10,facecolor='red', alpha=0.5)
# fig.subplots_adjust(left=0, right=1, bottom=0, top=0.5, hspace=0.05, wspace=1)
ax[0].set_ylim([0, maxx])
ax[0].set_xlabel("Responders "+col)
ax[0].set_ylabel("")
ax[1].set_ylim([0, maxx])
ax[1].set_xlabel("NonResponders "+col)
ax[1].set_ylabel("")
#ax[0].legend(loc='best')
#ax[1].legend(loc='best')
fig.suptitle("Distribution of "+col)
plt.show()

############### Correlation
label_encoder = LabelEncoder()
datadesc['Responder?'] = label_encoder.fit_transform(datadesc['Responder?'].astype(str))
datadesc['Initial dx'] = label_encoder.fit_transform(datadesc['Initial dx'].astype(str))
datadesc['Race'] = label_encoder.fit_transform(datadesc['Race'].astype(str))
datadesc['DM (Y/N)'] = label_encoder.fit_transform(datadesc['DM (Y/N)'].astype(str))
datadesc['Progestin Use (type/agent)'] = label_encoder.fit_transform(datadesc['Progestin Use (type/agent)'].astype(str))
datadesc['FHx of endometrial CA'] = label_encoder.fit_transform(datadesc['FHx of endometrial CA'].astype(str))
datadesc['PHx of breast/ovarian CA'] = label_encoder.fit_transform(datadesc['PHx of breast/ovarian CA'].astype(str))
print(datadesc['Responder?'].unique())

corr = datadesc.corr()
matrix = np.triu(corr)
sns.heatmap(datadesc.corr(), annot=True,vmax=1,center=0,cmap='coolwarm',mask=matrix)

sns.heatmap(corr,
            xticklabels=corr.columns.values,
            yticklabels=corr.columns.values)



corr = datadesc.corr(method='spearman')
sns.heatmap(corr,
            xticklabels=corr.columns.values,
            yticklabels=corr.columns.values)

##################### TWO VARIABLES - LINEAR MODEL PLOT
datadesc1 = datadesc

# 'BMI at follow-up (kg)'
col1 = 'Age at dx'
col2 = 'BMI at follow-up (kg)'
datadesc1=datadesc1.loc[datadesc1[col1].notnull()]
datadesc1=datadesc1.loc[datadesc1[col2].notnull()]
datadesc1[col1] = datadesc1[col1].astype(float).astype(int)
datadesc1[col2] = datadesc1[col2].astype(float).astype(int)
sns.lmplot(data=datadesc1,x=col1,y=col2)










######################################### Machine Learning
X = datadesc[['Initial dx',
       'Age at dx',
       'BMI at dx (kg)', 'BMI at follow-up (kg)', 'Race', 'DM (Y/N)',
       'Progestin Use (type/agent)', 'FHx of endometrial CA',
       'PHx of breast/ovarian CA']]
X = X.fillna(0)

y = datadesc[['Responder?']]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)



################                                      Normalize Data
# Define the scaler
scaler = StandardScaler().fit(X_train)
# Scale the train set
X_train = scaler.transform(X_train)
# Scale the test set
X_test = scaler.transform(X_test)

################################### Create the model
# Initialize the constructor
model = Sequential()  # comes from import: from keras.models import Sequential
# Add an input layer
model.add(Dense(12, activation='relu', input_shape=(9,)))
# Add one hidden layer
model.add(Dense(8, activation='relu'))
# Add an output layer
model.add(Dense(1, activation='sigmoid'))
# Model output shape
model.output_shape
# Model summary
model.summary()
# Model config
model.get_config()
# List all weight tensors
model.get_weights()


########## Compile and fit the Model
model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])
history = model.fit(X_train, y_train, epochs=200, batch_size=1, verbose=1)



########## list all data in history
print(history.history.keys())
# summarize history for accuracy
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

#######                Predict Values
y_pred = model.predict(X_test)
# to compare predict and test
y_pred[:5]
y_test[:5]
###############                            Evaluate Model
score = model.evaluate(X_test, y_test,verbose=1)
print(score)
###############  Confusion matrix

conf = confusion_matrix(y_test.round(), y_pred.round())
print("Confusion Matrix: ", conf)
###############  Precision
precision = precision_score(y_test.round(), y_pred.round()) #  average=Nonefor precision from each class
print("Precision: ",precision)
############### Recall
recall = recall_score(y_test.round(), y_pred.round())
print("Recall: ",recall)
############### F1 score
f1_score = f1_score(y_test.round(), y_pred.round())
print("F1 Score: ",f1_score)
############### Cohen's kappa
cohen_kappa_score = cohen_kappa_score(y_test.round(), y_pred.round())
print("Cohen_Kappa Score: ",cohen_kappa_score)

print("Done!")




