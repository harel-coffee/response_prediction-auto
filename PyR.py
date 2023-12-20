#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 28 09:27:45 2023

@author: seyedm.mousavikahaki
"""

from radiomics import featureextractor
import SimpleITK as sitk
import numpy as np
from PIL import Image
import pandas as pd
import shutil
import os
import radiomics
import logging
# Instantiate the extractor
extractor = featureextractor.RadiomicsFeatureExtractor(binWidth=20, sigma=[1, 2, 3], verbose=True)

# First define the parameters
params = {}
params['binWidth'] = 20
params['sigma'] = [1, 2, 3]
params['verbose'] = True

# Instantiate the extractor
extractor = featureextractor.RadiomicsFeatureExtractor(**params)  # ** 'unpacks' the dictionary in the function call


# Enable a filter (in addition to the 'Original' filter already enabled)
extractor.enableImageTypeByName('LoG')

# Disable all feature classes, save firstorder
extractor.disableAllFeatures()
extractor.enableFeatureClassByName('firstorder')
extractor.enableFeatureClassByName('glcm')
extractor.enableFeatureClassByName('glszm')
extractor.enableFeatureClassByName('glrlm')
extractor.enableFeatureClassByName('ngtdm')
extractor.enableFeatureClassByName('gldm')

radiomics.logger.setLevel(logging.ERROR)

### Training
import os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import pandas as pd

sourceIn = '/home/seyedm.mousavikahaki/Documents/response_prediction_data/png_files/Training/'
sourceIn_Mask = '/home/seyedm.mousavikahaki/Documents/response_prediction_data/png_files/Training_Mask/'

target = '/home/seyedm.mousavikahaki/Documents/response_prediction_data/Features/RadiometricFeatures/Training/'
allFolders= os.listdir(sourceIn) 
errorfiles = []
print(len(sourceIn))
errorfilenames = []

for Folder in allFolders:
    print(Folder)
    print("====================================================")
    files= os.fsencode(sourceIn+Folder)
    WSI_Features = []
    for file in os.listdir(files):
        try:
            filename = os.fsdecode(file)
            
            imagePath = sourceIn+Folder+'/'+filename
            
            imagePath = imagePath.replace('Training','Training_Mask').replace('.png','_gray.png')
            maskPath =  imagePath.replace('_gray.png','.nrrd')
            
            
            result = extractor.execute(imagePath,maskPath)
            
            WSI_Features.append(
            {
            'imagePath': imagePath,
            'original_firstorder_10Percentile': np.float(result['original_firstorder_10Percentile']),
            'original_firstorder_90Percentile': np.float(result['original_firstorder_90Percentile']),
            'original_firstorder_Energy': np.float(result['original_firstorder_Energy']),
            'original_firstorder_Entropy': np.float(result['original_firstorder_Entropy']),
            'original_firstorder_InterquartileRange': np.float(result['original_firstorder_InterquartileRange']),
            'original_firstorder_Kurtosis': np.float(result['original_firstorder_Kurtosis']),
            'original_firstorder_Maximum': np.float(result['original_firstorder_Maximum']),
            'original_firstorder_Mean': np.float(result['original_firstorder_Mean']),
            'original_firstorder_MeanAbsoluteDeviation': np.float(result['original_firstorder_MeanAbsoluteDeviation']),
            'original_firstorder_Median': np.float(result['original_firstorder_Median']),
            'original_firstorder_Minimum': np.float(result['original_firstorder_Minimum']),
            'original_firstorder_Range': np.float(result['original_firstorder_Range']),
            'original_firstorder_Median': np.float(result['original_firstorder_Median']),
            'original_firstorder_RobustMeanAbsoluteDeviation': np.float(result['original_firstorder_RobustMeanAbsoluteDeviation']),
            'original_firstorder_RootMeanSquared': np.float(result['original_firstorder_RootMeanSquared']),
            'original_firstorder_Skewness': np.float(result['original_firstorder_Skewness']),
            'original_firstorder_TotalEnergy': np.float(result['original_firstorder_TotalEnergy']),
            'original_firstorder_Uniformity': np.float(result['original_firstorder_Uniformity']),
            'original_firstorder_Variance': np.float(result['original_firstorder_Variance']),
            'original_glcm_Autocorrelation': np.float(result['original_glcm_Autocorrelation']),
            'original_glcm_ClusterProminence': np.float(result['original_glcm_ClusterProminence']),
            'original_glcm_ClusterShade': np.float(result['original_glcm_ClusterShade']),
            'original_glcm_ClusterTendency': np.float(result['original_glcm_ClusterTendency']),
            'original_glcm_Contrast': np.float(result['original_glcm_Contrast']),
            'original_glcm_Correlation': np.float(result['original_glcm_Correlation']),
            'original_glcm_DifferenceAverage': np.float(result['original_glcm_DifferenceAverage']),
            'original_glcm_DifferenceEntropy': np.float(result['original_glcm_DifferenceEntropy']),
            'original_glcm_DifferenceVariance': np.float(result['original_glcm_DifferenceVariance']),
            'original_glcm_Id': np.float(result['original_glcm_Id']),
            'original_glcm_Idm': np.float(result['original_glcm_Idm']),
            'original_glcm_Idmn': np.float(result['original_glcm_Idmn']),
            'original_glcm_Idn': np.float(result['original_glcm_Idn']),
            'original_glcm_Imc1': np.float(result['original_glcm_Imc1']),
            'original_glcm_Imc2': np.float(result['original_glcm_Imc2']),
            'original_glcm_InverseVariance': np.float(result['original_glcm_InverseVariance']),
            'original_glcm_JointAverage': np.float(result['original_glcm_JointAverage']),
            'original_glcm_JointEnergy': np.float(result['original_glcm_JointEnergy']),
            'original_glcm_JointEntropy': np.float(result['original_glcm_JointEntropy']),
            'original_glcm_MaximumProbability': np.float(result['original_glcm_MaximumProbability']),
            'original_glcm_MCC': np.float(result['original_glcm_MCC']),
            'original_glcm_SumAverage': np.float(result['original_glcm_SumAverage']),
            'original_glcm_SumEntropy': np.float(result['original_glcm_SumEntropy']),
            'original_glcm_SumSquares': np.float(result['original_glcm_SumSquares']),
            'original_gldm_DependenceEntropy': np.float(result['original_gldm_DependenceEntropy']),
            'original_gldm_DependenceNonUniformity': np.float(result['original_gldm_DependenceNonUniformity']),
            'original_gldm_DependenceNonUniformityNormalized': np.float(result['original_gldm_DependenceNonUniformityNormalized']),
            'original_gldm_DependenceVariance': np.float(result['original_gldm_DependenceVariance']),
            'original_gldm_GrayLevelNonUniformity': np.float(result['original_gldm_GrayLevelNonUniformity']),
            'original_gldm_GrayLevelVariance': np.float(result['original_gldm_GrayLevelVariance']),
            'original_gldm_HighGrayLevelEmphasis': np.float(result['original_gldm_HighGrayLevelEmphasis']),
            'original_gldm_LargeDependenceEmphasis': np.float(result['original_gldm_LargeDependenceEmphasis']),
            'original_gldm_LargeDependenceHighGrayLevelEmphasis': np.float(result['original_gldm_LargeDependenceHighGrayLevelEmphasis']),
            'original_gldm_LargeDependenceLowGrayLevelEmphasis': np.float(result['original_gldm_LargeDependenceLowGrayLevelEmphasis']),
            'original_gldm_LowGrayLevelEmphasis': np.float(result['original_gldm_LowGrayLevelEmphasis']),
            'original_gldm_SmallDependenceEmphasis': np.float(result['original_gldm_SmallDependenceEmphasis']),
            'original_gldm_SmallDependenceHighGrayLevelEmphasis': np.float(result['original_gldm_SmallDependenceHighGrayLevelEmphasis']),
            'original_gldm_SmallDependenceLowGrayLevelEmphasis': np.float(result['original_gldm_SmallDependenceLowGrayLevelEmphasis']),
            'original_glrlm_GrayLevelNonUniformity': np.float(result['original_glrlm_GrayLevelNonUniformity']),
            'original_glrlm_GrayLevelNonUniformityNormalized': np.float(result['original_glrlm_GrayLevelNonUniformityNormalized']),
            'original_glrlm_GrayLevelVariance': np.float(result['original_glrlm_GrayLevelVariance']),
            'original_glrlm_HighGrayLevelRunEmphasis': np.float(result['original_glrlm_HighGrayLevelRunEmphasis']),
            'original_glrlm_LongRunEmphasis': np.float(result['original_glrlm_LongRunEmphasis']),
            'original_glrlm_LongRunHighGrayLevelEmphasis': np.float(result['original_glrlm_LongRunHighGrayLevelEmphasis']),
            'original_glrlm_LongRunLowGrayLevelEmphasis': np.float(result['original_glrlm_LongRunLowGrayLevelEmphasis']),
            'original_glrlm_LowGrayLevelRunEmphasis': np.float(result['original_glrlm_LowGrayLevelRunEmphasis']),
            'original_glrlm_RunEntropy': np.float(result['original_glrlm_RunEntropy']),
            'original_glrlm_RunLengthNonUniformity': np.float(result['original_glrlm_RunLengthNonUniformity']),
            'original_glrlm_RunLengthNonUniformityNormalized': np.float(result['original_glrlm_RunLengthNonUniformityNormalized']),
            'original_glrlm_RunPercentage': np.float(result['original_glrlm_RunPercentage']),
            'original_glrlm_RunVariance': np.float(result['original_glrlm_RunVariance']),
            'original_glrlm_ShortRunEmphasis': np.float(result['original_glrlm_ShortRunEmphasis']),
            'original_glrlm_ShortRunHighGrayLevelEmphasis': np.float(result['original_glrlm_ShortRunHighGrayLevelEmphasis']),
            'original_glrlm_ShortRunLowGrayLevelEmphasis': np.float(result['original_glrlm_ShortRunLowGrayLevelEmphasis']),
            'original_glszm_GrayLevelNonUniformity': np.float(result['original_glszm_GrayLevelNonUniformity']),
            'original_glszm_GrayLevelNonUniformityNormalized': np.float(result['original_glszm_GrayLevelNonUniformityNormalized']),
            'original_glszm_GrayLevelVariance': np.float(result['original_glszm_GrayLevelVariance']),
            'original_glszm_HighGrayLevelZoneEmphasis': np.float(result['original_glszm_HighGrayLevelZoneEmphasis']),
            'original_glszm_LargeAreaEmphasis': np.float(result['original_glszm_LargeAreaEmphasis']),
            'original_glszm_LargeAreaHighGrayLevelEmphasis': np.float(result['original_glszm_LargeAreaHighGrayLevelEmphasis']),
            'original_glszm_LargeAreaLowGrayLevelEmphasis': np.float(result['original_glszm_LargeAreaLowGrayLevelEmphasis']),
            'original_glszm_LowGrayLevelZoneEmphasis': np.float(result['original_glszm_LowGrayLevelZoneEmphasis']),
            'original_glszm_SizeZoneNonUniformity': np.float(result['original_glszm_SizeZoneNonUniformity']),
            'original_glszm_SizeZoneNonUniformityNormalized': np.float(result['original_glszm_SizeZoneNonUniformityNormalized']),
            'original_glszm_SmallAreaEmphasis': np.float(result['original_glszm_SmallAreaEmphasis']),
            'original_glszm_SmallAreaHighGrayLevelEmphasis': np.float(result['original_glszm_SmallAreaHighGrayLevelEmphasis']),
            'original_glszm_SmallAreaLowGrayLevelEmphasis': np.float(result['original_glszm_SmallAreaLowGrayLevelEmphasis']),
            'original_glszm_ZoneEntropy': np.float(result['original_glszm_ZoneEntropy']),
            'original_glszm_ZonePercentage': np.float(result['original_glszm_ZonePercentage']),
            'original_glszm_ZoneVariance': np.float(result['original_glszm_ZoneVariance']),
            'original_ngtdm_Busyness': np.float(result['original_ngtdm_Busyness']),
            'original_ngtdm_Coarseness': np.float(result['original_ngtdm_Coarseness']),
            'original_ngtdm_Complexity': np.float(result['original_ngtdm_Complexity']),
            'original_ngtdm_Contrast': np.float(result['original_ngtdm_Contrast']),
            'original_ngtdm_Strength': np.float(result['original_ngtdm_Strength'])
            
            })
         
        except:
            errorfiles.append(imagePath)
            print("______________ERROR___________")
            print(imagePath)
    WSI_Features = pd.DataFrame(WSI_Features)
    WSI_Features.to_csv(sourceIn.replace('Training','Training_FeaturesNEW')+Folder+'.csv')
    
    print('error Files')
    print(errorfiles)

### Testing
from PIL import Image
import matplotlib.pyplot as plt
import pandas as pd
sourceIn = '/home/seyedm.mousavikahaki/Documents/response_prediction_data/png_files/Testing/'
sourceIn_Mask = '/home/seyedm.mousavikahaki/Documents/response_prediction_data/png_files/Testing_Mask/'

allFolders= os.listdir(sourceIn) 
errorfiles = []
print(len(sourceIn))
errorfilenames = []

for Folder in allFolders:
    print(Folder)
    files= os.fsencode(sourceIn+Folder)
    WSI_Features = []
    for file in os.listdir(files):
        try:
            filename = os.fsdecode(file)
            print(filename)
        
            imagePath = sourceIn+Folder+'/'+filename

            imagePath = imagePath.replace('Testing','Testing_Mask').replace('.png','_gray.png')
            maskPath =  imagePath.replace('_gray.png','.nrrd')
  
            result = extractor.execute(imagePath,maskPath)
            
            WSI_Features.append(
            {
            'imagePath': imagePath,
            'original_firstorder_10Percentile': np.float(result['original_firstorder_10Percentile']),
            'original_firstorder_90Percentile': np.float(result['original_firstorder_90Percentile']),
            'original_firstorder_Energy': np.float(result['original_firstorder_Energy']),
            'original_firstorder_Entropy': np.float(result['original_firstorder_Entropy']),
            'original_firstorder_InterquartileRange': np.float(result['original_firstorder_InterquartileRange']),
            'original_firstorder_Kurtosis': np.float(result['original_firstorder_Kurtosis']),
            'original_firstorder_Maximum': np.float(result['original_firstorder_Maximum']),
            'original_firstorder_Mean': np.float(result['original_firstorder_Mean']),
            'original_firstorder_MeanAbsoluteDeviation': np.float(result['original_firstorder_MeanAbsoluteDeviation']),
            'original_firstorder_Median': np.float(result['original_firstorder_Median']),
            'original_firstorder_Minimum': np.float(result['original_firstorder_Minimum']),
            'original_firstorder_Range': np.float(result['original_firstorder_Range']),
            'original_firstorder_Median': np.float(result['original_firstorder_Median']),
            'original_firstorder_RobustMeanAbsoluteDeviation': np.float(result['original_firstorder_RobustMeanAbsoluteDeviation']),
            'original_firstorder_RootMeanSquared': np.float(result['original_firstorder_RootMeanSquared']),
            'original_firstorder_Skewness': np.float(result['original_firstorder_Skewness']),
            'original_firstorder_TotalEnergy': np.float(result['original_firstorder_TotalEnergy']),
            'original_firstorder_Uniformity': np.float(result['original_firstorder_Uniformity']),
            'original_firstorder_Variance': np.float(result['original_firstorder_Variance']),
            'original_glcm_Autocorrelation': np.float(result['original_glcm_Autocorrelation']),
            'original_glcm_ClusterProminence': np.float(result['original_glcm_ClusterProminence']),
            'original_glcm_ClusterShade': np.float(result['original_glcm_ClusterShade']),
            'original_glcm_ClusterTendency': np.float(result['original_glcm_ClusterTendency']),
            'original_glcm_Contrast': np.float(result['original_glcm_Contrast']),
            'original_glcm_Correlation': np.float(result['original_glcm_Correlation']),
            'original_glcm_DifferenceAverage': np.float(result['original_glcm_DifferenceAverage']),
            'original_glcm_DifferenceEntropy': np.float(result['original_glcm_DifferenceEntropy']),
            'original_glcm_DifferenceVariance': np.float(result['original_glcm_DifferenceVariance']),
            'original_glcm_Id': np.float(result['original_glcm_Id']),
            'original_glcm_Idm': np.float(result['original_glcm_Idm']),
            'original_glcm_Idmn': np.float(result['original_glcm_Idmn']),
            'original_glcm_Idn': np.float(result['original_glcm_Idn']),
            'original_glcm_Imc1': np.float(result['original_glcm_Imc1']),
            'original_glcm_Imc2': np.float(result['original_glcm_Imc2']),
            'original_glcm_InverseVariance': np.float(result['original_glcm_InverseVariance']),
            'original_glcm_JointAverage': np.float(result['original_glcm_JointAverage']),
            'original_glcm_JointEnergy': np.float(result['original_glcm_JointEnergy']),
            'original_glcm_JointEntropy': np.float(result['original_glcm_JointEntropy']),
            'original_glcm_MaximumProbability': np.float(result['original_glcm_MaximumProbability']),
            'original_glcm_MCC': np.float(result['original_glcm_MCC']),
            'original_glcm_SumAverage': np.float(result['original_glcm_SumAverage']),
            'original_glcm_SumEntropy': np.float(result['original_glcm_SumEntropy']),
            'original_glcm_SumSquares': np.float(result['original_glcm_SumSquares']),
            'original_gldm_DependenceEntropy': np.float(result['original_gldm_DependenceEntropy']),
            'original_gldm_DependenceNonUniformity': np.float(result['original_gldm_DependenceNonUniformity']),
            'original_gldm_DependenceNonUniformityNormalized': np.float(result['original_gldm_DependenceNonUniformityNormalized']),
            'original_gldm_DependenceVariance': np.float(result['original_gldm_DependenceVariance']),
            'original_gldm_GrayLevelNonUniformity': np.float(result['original_gldm_GrayLevelNonUniformity']),
            'original_gldm_GrayLevelVariance': np.float(result['original_gldm_GrayLevelVariance']),
            'original_gldm_HighGrayLevelEmphasis': np.float(result['original_gldm_HighGrayLevelEmphasis']),
            'original_gldm_LargeDependenceEmphasis': np.float(result['original_gldm_LargeDependenceEmphasis']),
            'original_gldm_LargeDependenceHighGrayLevelEmphasis': np.float(result['original_gldm_LargeDependenceHighGrayLevelEmphasis']),
            'original_gldm_LargeDependenceLowGrayLevelEmphasis': np.float(result['original_gldm_LargeDependenceLowGrayLevelEmphasis']),
            'original_gldm_LowGrayLevelEmphasis': np.float(result['original_gldm_LowGrayLevelEmphasis']),
            'original_gldm_SmallDependenceEmphasis': np.float(result['original_gldm_SmallDependenceEmphasis']),
            'original_gldm_SmallDependenceHighGrayLevelEmphasis': np.float(result['original_gldm_SmallDependenceHighGrayLevelEmphasis']),
            'original_gldm_SmallDependenceLowGrayLevelEmphasis': np.float(result['original_gldm_SmallDependenceLowGrayLevelEmphasis']),
            'original_glrlm_GrayLevelNonUniformity': np.float(result['original_glrlm_GrayLevelNonUniformity']),
            'original_glrlm_GrayLevelNonUniformityNormalized': np.float(result['original_glrlm_GrayLevelNonUniformityNormalized']),
            'original_glrlm_GrayLevelVariance': np.float(result['original_glrlm_GrayLevelVariance']),
            'original_glrlm_HighGrayLevelRunEmphasis': np.float(result['original_glrlm_HighGrayLevelRunEmphasis']),
            'original_glrlm_LongRunEmphasis': np.float(result['original_glrlm_LongRunEmphasis']),
            'original_glrlm_LongRunHighGrayLevelEmphasis': np.float(result['original_glrlm_LongRunHighGrayLevelEmphasis']),
            'original_glrlm_LongRunLowGrayLevelEmphasis': np.float(result['original_glrlm_LongRunLowGrayLevelEmphasis']),
            'original_glrlm_LowGrayLevelRunEmphasis': np.float(result['original_glrlm_LowGrayLevelRunEmphasis']),
            'original_glrlm_RunEntropy': np.float(result['original_glrlm_RunEntropy']),
            'original_glrlm_RunLengthNonUniformity': np.float(result['original_glrlm_RunLengthNonUniformity']),
            'original_glrlm_RunLengthNonUniformityNormalized': np.float(result['original_glrlm_RunLengthNonUniformityNormalized']),
            'original_glrlm_RunPercentage': np.float(result['original_glrlm_RunPercentage']),
            'original_glrlm_RunVariance': np.float(result['original_glrlm_RunVariance']),
            'original_glrlm_ShortRunEmphasis': np.float(result['original_glrlm_ShortRunEmphasis']),
            'original_glrlm_ShortRunHighGrayLevelEmphasis': np.float(result['original_glrlm_ShortRunHighGrayLevelEmphasis']),
            'original_glrlm_ShortRunLowGrayLevelEmphasis': np.float(result['original_glrlm_ShortRunLowGrayLevelEmphasis']),
            'original_glszm_GrayLevelNonUniformity': np.float(result['original_glszm_GrayLevelNonUniformity']),
            'original_glszm_GrayLevelNonUniformityNormalized': np.float(result['original_glszm_GrayLevelNonUniformityNormalized']),
            'original_glszm_GrayLevelVariance': np.float(result['original_glszm_GrayLevelVariance']),
            'original_glszm_HighGrayLevelZoneEmphasis': np.float(result['original_glszm_HighGrayLevelZoneEmphasis']),
            'original_glszm_LargeAreaEmphasis': np.float(result['original_glszm_LargeAreaEmphasis']),
            'original_glszm_LargeAreaHighGrayLevelEmphasis': np.float(result['original_glszm_LargeAreaHighGrayLevelEmphasis']),
            'original_glszm_LargeAreaLowGrayLevelEmphasis': np.float(result['original_glszm_LargeAreaLowGrayLevelEmphasis']),
            'original_glszm_LowGrayLevelZoneEmphasis': np.float(result['original_glszm_LowGrayLevelZoneEmphasis']),
            'original_glszm_SizeZoneNonUniformity': np.float(result['original_glszm_SizeZoneNonUniformity']),
            'original_glszm_SizeZoneNonUniformityNormalized': np.float(result['original_glszm_SizeZoneNonUniformityNormalized']),
            'original_glszm_SmallAreaEmphasis': np.float(result['original_glszm_SmallAreaEmphasis']),
            'original_glszm_SmallAreaHighGrayLevelEmphasis': np.float(result['original_glszm_SmallAreaHighGrayLevelEmphasis']),
            'original_glszm_SmallAreaLowGrayLevelEmphasis': np.float(result['original_glszm_SmallAreaLowGrayLevelEmphasis']),
            'original_glszm_ZoneEntropy': np.float(result['original_glszm_ZoneEntropy']),
            'original_glszm_ZonePercentage': np.float(result['original_glszm_ZonePercentage']),
            'original_glszm_ZoneVariance': np.float(result['original_glszm_ZoneVariance']),
            'original_ngtdm_Busyness': np.float(result['original_ngtdm_Busyness']),
            'original_ngtdm_Coarseness': np.float(result['original_ngtdm_Coarseness']),
            'original_ngtdm_Complexity': np.float(result['original_ngtdm_Complexity']),
            'original_ngtdm_Contrast': np.float(result['original_ngtdm_Contrast']),
            'original_ngtdm_Strength': np.float(result['original_ngtdm_Strength'])
            
            })
            

            
        except:
            errorfiles.append(imagePath)
    WSI_Features = pd.DataFrame(WSI_Features)
    WSI_Features.to_csv(sourceIn.replace('Testing','Testing_FeaturesNEW')+Folder+'.csv')
    
    print('error Files')
    print(errorfiles)



### Tunning
from PIL import Image
import matplotlib.pyplot as plt
sourceIn = '/home/seyedm.mousavikahaki/Documents/response_prediction_data/png_files/Tunning/'
sourceIn_Mask = '/home/seyedm.mousavikahaki/Documents/response_prediction_data/png_files/Tunning_Mask/'

allFolders= os.listdir(sourceIn) 
errorfiles = []
print(len(sourceIn))
errorfilenames = []

for Folder in allFolders:
    print(Folder)
    files= os.fsencode(sourceIn+Folder)
    WSI_Features = []
    for file in os.listdir(files):
        try:
            filename = os.fsdecode(file)
            print(filename)
        
            imagePath = sourceIn+Folder+'/'+filename           
            
            imagePath = imagePath.replace('Tunning','Tunning_Mask').replace('.png','_gray.png')
            maskPath =  imagePath.replace('_gray.png','.nrrd')
            
            result = extractor.execute(imagePath,maskPath)
            
            WSI_Features.append(
            {
            'imagePath': imagePath,
            'original_firstorder_10Percentile': np.float(result['original_firstorder_10Percentile']),
            'original_firstorder_90Percentile': np.float(result['original_firstorder_90Percentile']),
            'original_firstorder_Energy': np.float(result['original_firstorder_Energy']),
            'original_firstorder_Entropy': np.float(result['original_firstorder_Entropy']),
            'original_firstorder_InterquartileRange': np.float(result['original_firstorder_InterquartileRange']),
            'original_firstorder_Kurtosis': np.float(result['original_firstorder_Kurtosis']),
            'original_firstorder_Maximum': np.float(result['original_firstorder_Maximum']),
            'original_firstorder_Mean': np.float(result['original_firstorder_Mean']),
            'original_firstorder_MeanAbsoluteDeviation': np.float(result['original_firstorder_MeanAbsoluteDeviation']),
            'original_firstorder_Median': np.float(result['original_firstorder_Median']),
            'original_firstorder_Minimum': np.float(result['original_firstorder_Minimum']),
            'original_firstorder_Range': np.float(result['original_firstorder_Range']),
            'original_firstorder_Median': np.float(result['original_firstorder_Median']),
            'original_firstorder_RobustMeanAbsoluteDeviation': np.float(result['original_firstorder_RobustMeanAbsoluteDeviation']),
            'original_firstorder_RootMeanSquared': np.float(result['original_firstorder_RootMeanSquared']),
            'original_firstorder_Skewness': np.float(result['original_firstorder_Skewness']),
            'original_firstorder_TotalEnergy': np.float(result['original_firstorder_TotalEnergy']),
            'original_firstorder_Uniformity': np.float(result['original_firstorder_Uniformity']),
            'original_firstorder_Variance': np.float(result['original_firstorder_Variance']),
            'original_glcm_Autocorrelation': np.float(result['original_glcm_Autocorrelation']),
            'original_glcm_ClusterProminence': np.float(result['original_glcm_ClusterProminence']),
            'original_glcm_ClusterShade': np.float(result['original_glcm_ClusterShade']),
            'original_glcm_ClusterTendency': np.float(result['original_glcm_ClusterTendency']),
            'original_glcm_Contrast': np.float(result['original_glcm_Contrast']),
            'original_glcm_Correlation': np.float(result['original_glcm_Correlation']),
            'original_glcm_DifferenceAverage': np.float(result['original_glcm_DifferenceAverage']),
            'original_glcm_DifferenceEntropy': np.float(result['original_glcm_DifferenceEntropy']),
            'original_glcm_DifferenceVariance': np.float(result['original_glcm_DifferenceVariance']),
            'original_glcm_Id': np.float(result['original_glcm_Id']),
            'original_glcm_Idm': np.float(result['original_glcm_Idm']),
            'original_glcm_Idmn': np.float(result['original_glcm_Idmn']),
            'original_glcm_Idn': np.float(result['original_glcm_Idn']),
            'original_glcm_Imc1': np.float(result['original_glcm_Imc1']),
            'original_glcm_Imc2': np.float(result['original_glcm_Imc2']),
            'original_glcm_InverseVariance': np.float(result['original_glcm_InverseVariance']),
            'original_glcm_JointAverage': np.float(result['original_glcm_JointAverage']),
            'original_glcm_JointEnergy': np.float(result['original_glcm_JointEnergy']),
            'original_glcm_JointEntropy': np.float(result['original_glcm_JointEntropy']),
            'original_glcm_MaximumProbability': np.float(result['original_glcm_MaximumProbability']),
            'original_glcm_MCC': np.float(result['original_glcm_MCC']),
            'original_glcm_SumAverage': np.float(result['original_glcm_SumAverage']),
            'original_glcm_SumEntropy': np.float(result['original_glcm_SumEntropy']),
            'original_glcm_SumSquares': np.float(result['original_glcm_SumSquares']),
            'original_gldm_DependenceEntropy': np.float(result['original_gldm_DependenceEntropy']),
            'original_gldm_DependenceNonUniformity': np.float(result['original_gldm_DependenceNonUniformity']),
            'original_gldm_DependenceNonUniformityNormalized': np.float(result['original_gldm_DependenceNonUniformityNormalized']),
            'original_gldm_DependenceVariance': np.float(result['original_gldm_DependenceVariance']),
            'original_gldm_GrayLevelNonUniformity': np.float(result['original_gldm_GrayLevelNonUniformity']),
            'original_gldm_GrayLevelVariance': np.float(result['original_gldm_GrayLevelVariance']),
            'original_gldm_HighGrayLevelEmphasis': np.float(result['original_gldm_HighGrayLevelEmphasis']),
            'original_gldm_LargeDependenceEmphasis': np.float(result['original_gldm_LargeDependenceEmphasis']),
            'original_gldm_LargeDependenceHighGrayLevelEmphasis': np.float(result['original_gldm_LargeDependenceHighGrayLevelEmphasis']),
            'original_gldm_LargeDependenceLowGrayLevelEmphasis': np.float(result['original_gldm_LargeDependenceLowGrayLevelEmphasis']),
            'original_gldm_LowGrayLevelEmphasis': np.float(result['original_gldm_LowGrayLevelEmphasis']),
            'original_gldm_SmallDependenceEmphasis': np.float(result['original_gldm_SmallDependenceEmphasis']),
            'original_gldm_SmallDependenceHighGrayLevelEmphasis': np.float(result['original_gldm_SmallDependenceHighGrayLevelEmphasis']),
            'original_gldm_SmallDependenceLowGrayLevelEmphasis': np.float(result['original_gldm_SmallDependenceLowGrayLevelEmphasis']),
            'original_glrlm_GrayLevelNonUniformity': np.float(result['original_glrlm_GrayLevelNonUniformity']),
            'original_glrlm_GrayLevelNonUniformityNormalized': np.float(result['original_glrlm_GrayLevelNonUniformityNormalized']),
            'original_glrlm_GrayLevelVariance': np.float(result['original_glrlm_GrayLevelVariance']),
            'original_glrlm_HighGrayLevelRunEmphasis': np.float(result['original_glrlm_HighGrayLevelRunEmphasis']),
            'original_glrlm_LongRunEmphasis': np.float(result['original_glrlm_LongRunEmphasis']),
            'original_glrlm_LongRunHighGrayLevelEmphasis': np.float(result['original_glrlm_LongRunHighGrayLevelEmphasis']),
            'original_glrlm_LongRunLowGrayLevelEmphasis': np.float(result['original_glrlm_LongRunLowGrayLevelEmphasis']),
            'original_glrlm_LowGrayLevelRunEmphasis': np.float(result['original_glrlm_LowGrayLevelRunEmphasis']),
            'original_glrlm_RunEntropy': np.float(result['original_glrlm_RunEntropy']),
            'original_glrlm_RunLengthNonUniformity': np.float(result['original_glrlm_RunLengthNonUniformity']),
            'original_glrlm_RunLengthNonUniformityNormalized': np.float(result['original_glrlm_RunLengthNonUniformityNormalized']),
            'original_glrlm_RunPercentage': np.float(result['original_glrlm_RunPercentage']),
            'original_glrlm_RunVariance': np.float(result['original_glrlm_RunVariance']),
            'original_glrlm_ShortRunEmphasis': np.float(result['original_glrlm_ShortRunEmphasis']),
            'original_glrlm_ShortRunHighGrayLevelEmphasis': np.float(result['original_glrlm_ShortRunHighGrayLevelEmphasis']),
            'original_glrlm_ShortRunLowGrayLevelEmphasis': np.float(result['original_glrlm_ShortRunLowGrayLevelEmphasis']),
            'original_glszm_GrayLevelNonUniformity': np.float(result['original_glszm_GrayLevelNonUniformity']),
            'original_glszm_GrayLevelNonUniformityNormalized': np.float(result['original_glszm_GrayLevelNonUniformityNormalized']),
            'original_glszm_GrayLevelVariance': np.float(result['original_glszm_GrayLevelVariance']),
            'original_glszm_HighGrayLevelZoneEmphasis': np.float(result['original_glszm_HighGrayLevelZoneEmphasis']),
            'original_glszm_LargeAreaEmphasis': np.float(result['original_glszm_LargeAreaEmphasis']),
            'original_glszm_LargeAreaHighGrayLevelEmphasis': np.float(result['original_glszm_LargeAreaHighGrayLevelEmphasis']),
            'original_glszm_LargeAreaLowGrayLevelEmphasis': np.float(result['original_glszm_LargeAreaLowGrayLevelEmphasis']),
            'original_glszm_LowGrayLevelZoneEmphasis': np.float(result['original_glszm_LowGrayLevelZoneEmphasis']),
            'original_glszm_SizeZoneNonUniformity': np.float(result['original_glszm_SizeZoneNonUniformity']),
            'original_glszm_SizeZoneNonUniformityNormalized': np.float(result['original_glszm_SizeZoneNonUniformityNormalized']),
            'original_glszm_SmallAreaEmphasis': np.float(result['original_glszm_SmallAreaEmphasis']),
            'original_glszm_SmallAreaHighGrayLevelEmphasis': np.float(result['original_glszm_SmallAreaHighGrayLevelEmphasis']),
            'original_glszm_SmallAreaLowGrayLevelEmphasis': np.float(result['original_glszm_SmallAreaLowGrayLevelEmphasis']),
            'original_glszm_ZoneEntropy': np.float(result['original_glszm_ZoneEntropy']),
            'original_glszm_ZonePercentage': np.float(result['original_glszm_ZonePercentage']),
            'original_glszm_ZoneVariance': np.float(result['original_glszm_ZoneVariance']),
            'original_ngtdm_Busyness': np.float(result['original_ngtdm_Busyness']),
            'original_ngtdm_Coarseness': np.float(result['original_ngtdm_Coarseness']),
            'original_ngtdm_Complexity': np.float(result['original_ngtdm_Complexity']),
            'original_ngtdm_Contrast': np.float(result['original_ngtdm_Contrast']),
            'original_ngtdm_Strength': np.float(result['original_ngtdm_Strength'])
            
            })
            
        except:
            errorfiles.append(imagePath)
    WSI_Features = pd.DataFrame(WSI_Features)
    WSI_Features.to_csv(sourceIn.replace('Tunning','Tunning_FeaturesNEW')+Folder+'.csv')
    
    print('error Files')
    print(errorfiles)



sourceIn = '/home/seyedm.mousavikahaki/Documents/response_prediction_data/png_files/Training/'
target = '/home/seyedm.mousavikahaki/Documents/response_prediction_data/png_files/Training_Mask/'
allFolders= os.listdir(sourceIn)
errorfiles = []
print(len(sourceIn))
errorfilenames = []
for Folder in allFolders:
    print(Folder)
    files= os.fsencode(sourceIn+Folder)
    for file in os.listdir(files):
        try:
            filename = os.fsdecode(file)
            print(filename)
        
            imagePath = sourceIn+Folder+'/'+filename
    
            im = sitk.ReadImage(imagePath)
            point = (10, 10)  
            roi_size = (250,250)  
            im_size = im.GetSize()[::-1]  
         
            ma_arr = np.zeros(im_size)
            
            ma_arr[1:255, 1:255] = 1

            ma = sitk.GetImageFromArray(ma_arr)
            ma.CopyInformation(im)  
            targetPath = target+Folder

            if not os.path.exists(targetPath):
                os.makedirs(targetPath)
            imgG = Image.open(imagePath).convert('L')
            imgG.save(targetPath+'/'+filename.replace('.png', '_gray.png'))
        
        except:
            errorfiles.append(imagePath)
       

sourceIn = '/home/seyedm.mousavikahaki/Documents/response_prediction_data/png_files/Tunning/'
target = '/home/seyedm.mousavikahaki/Documents/response_prediction_data/png_files/Tunning_Mask/'
allFolders= os.listdir(sourceIn) # Training refers to the Response prediction Training, just response prediction training used here
print(len(sourceIn))
errorfilenames = []
for Folder in allFolders:
    print(Folder)
    files= os.fsencode(sourceIn+Folder)
    for file in os.listdir(files):
        try:
            filename = os.fsdecode(file)
            print(filename)
        
            imagePath = sourceIn+Folder+'/'+filename
    
            im = sitk.ReadImage(imagePath)
            point = (10, 10)  
            roi_size = (250,250) 
            
            im_size = im.GetSize()[::-1]  
            
            ma_arr = np.zeros(im_size)
            
            ma_arr[1:255, 1:255] = 1

            ma = sitk.GetImageFromArray(ma_arr)
            ma.CopyInformation(im)  # This copies the geometric information, ensuring image and mask are aligned. This works, because image and mask have the same size of the pixel array
            targetPath = target+Folder
            if not os.path.exists(targetPath):
                os.makedirs(targetPath)

            imgG = Image.open(imagePath).convert('L')
            imgG.save(targetPath+'/'+filename.replace('.png', '_gray.png'))
        
       
        except:
            errorfiles.append(imagePath)
            

sourceIn = '/home/seyedm.mousavikahaki/Documents/response_prediction_data/png_files/Testing/'
target = '/home/seyedm.mousavikahaki/Documents/response_prediction_data/png_files/Testing_Mask/'
allFolders= os.listdir(sourceIn) 
print(len(sourceIn))
errorfilenames = []
for Folder in allFolders:
    print(Folder)
    files= os.fsencode(sourceIn+Folder)
    for file in os.listdir(files):
        try:
            filename = os.fsdecode(file)
            print(filename)
        
            imagePath = sourceIn+Folder+'/'+filename
    
            im = sitk.ReadImage(imagePath)
            point = (10, 10) 
            roi_size = (250,250) 
            
            im_size = im.GetSize()[::-1] 
            
            ma_arr = np.zeros(im_size)
            
            ma_arr[1:255, 1:255] = 1
            
            
            ma = sitk.GetImageFromArray(ma_arr)
            ma.CopyInformation(im)  
            targetPath = target+Folder
            if not os.path.exists(targetPath):
                os.makedirs(targetPath)
            
            imgG = Image.open(imagePath).convert('L')
            imgG.save(targetPath+'/'+filename.replace('.png', '_gray.png'))
        except:
            errorfiles.append(imagePath)




















