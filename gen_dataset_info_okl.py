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
# Description:  This code Split the data into training and test subsets
#               by stratification over several variables (columns)
#
#
# Input:        CSV file: Dataset 
# Output:       CSV file: training and test subsets
#
# 
# Example:      train_test_split.py
#
#
# version ='3.0'
# ---------------------------------------------------------------------------
"""
Created on Sat Sep 18 07:48:39 2021

@author: SeyedM.MousaviKahaki
"""

# import numpy as np
import pandas as pd
# import seaborn as sns
import matplotlib.pyplot as plt
# from sklearn import datasets, svm
import slideio
import glob
import xmltodict
from xml.dom import minidom
import numpy as np
import os
# import openslide
# from openslide import open_slide  
# from openslide.deepzoom import DeepZoomGenerator
import re
# import os
import seaborn as sns

if os.name == 'nt':
    print("Running on Windows!")
    ROOTPATH = 'C:/DATA/'
else:
    print("Running on Ubuntu!")  
    ROOTPATH = '/gpfs_projects/wxc4/DigiPath-WashU-Data/'    

datasetName = 'Aperio_Oklahoma' # 'Aperio' '3Dhistech' 'Aperio_Oklahoma' '3DHistech_Oklahoma'

ver = 'v1'
if datasetName == 'Aperio':
    DATAPATH = 'Washu-Aperio/'
elif datasetName == 'Aperio_Oklahoma':
    DATAPATH = 'Oklahoma-Aperio/'
elif datasetName == '3DHistech_Oklahoma':
    DATAPATH = 'Oklahoma-3DHistech/'   
else:
    DATAPATH = 'WashU-3DHistech/'


# Returns a list of names in list files.
print("Using glob.glob() to read svs files")
files = glob.glob(ROOTPATH+DATAPATH+'*.svs', recursive = True)


DATAFILE = 'C:/DATA/Master patient list for FDA 3-4-22_Seyed.csv'
############### READ DATA
data = pd.read_csv(DATAFILE)


####### Count SVS file differences - Not valid for Oklahoma Since names are not following same format
FileExistIn3DHist = 0
misscount = 0
missingHistechs = []
tmp = []
# Read svs files
for file in files:
    print("--------------------------------")
    print('ReadingSVS: ' + file)
    SVSName = os.path.basename(file)
    print()
    if datasetName == 'Aperio' or datasetName == 'Aperio_Oklahoma':
        
        PatiendID = data.loc[data['Filename of initial Aperio slide'] == SVSName]['Patient ID'].item()
        HistechFileName = data.loc[data['Filename of initial Aperio slide'] == SVSName]['Filename of initial 3D Histech slide'].item()
        
        print("Patient ID: "+str(PatiendID))
        timePoint = 0#int(file[len(ROOTPATH+DATAPATH+'aperio-'+strFilenum+'-'):-len('.svs')])
        Responder = data.loc[data['Filename of initial Aperio slide'] == SVSName]['Responder?'].item()
        print(HistechFileName)
        path3Dhist = ROOTPATH + 'WashU-3DHistech/' + str(HistechFileName)
        path3DhistOklahoma = ROOTPATH + 'Oklahoma-3DHistech/' + str(HistechFileName)
        
        FileExistIn3DHist = (os.path.exists(path3Dhist)|os.path.exists(path3DhistOklahoma))
        print(FileExistIn3DHist)
        if FileExistIn3DHist != True:
            misscount = misscount +1
            missingHistechs.append(SVSName)
    else:
        PatiendID = data.loc[data['Filename of initial 3D Histech slide'] == SVSName]['Patient ID'].item()
        print("Patient ID: "+str(PatiendID))
        FileExistIn3DHist = 0
        timePoint = 0
        Responder = 'NA'
    
    
    
    
    SVSFileSize = round(os.path.getsize(file) / (1024*1024*1024),2)
    xmlpath = str.replace(file, 'svs', 'xml')
    xmlexist = os.path.exists(xmlpath)
    
    
    # # # # # # # # # # # # # # # # # # # Read SVS file
    WSI = slideio.open_slide(file, 'SVS') 
    scene = WSI.get_scene(0)
    row_string = WSI.raw_metadata
    
    SVS_MetaData = row_string.split("|")
    if datasetName == 'Aperio':
        if PatiendID == 27:
            print(SVS_MetaData)
    
    # Get APPMag from MetaData
    SVS_Magnification = []
    given_string = row_string
    start_string = 'AppMag = '
    if datasetName == '3DHistech_Oklahoma':
        end_string = '|MPP' 
    else:
        end_string = '|StripeWidth'
    start_index = given_string.find(start_string) + len(start_string)
    end_index = given_string.find(end_string)

    SVS_Magnification = given_string[start_index:end_index]
    # Get MPP from MetaData
    SVS_MPP = []
    start_string = 'MPP = '
    end_string = '|Left ='
    start_index = given_string.find(start_string) + len(start_string)
    end_index = given_string.find(end_string)
    
    
    if datasetName == '3DHistech_Oklahoma':
        SVS_MPP = given_string[start_index:]
    else:
        SVS_MPP = given_string[start_index:end_index]
        
    
    SVS_num_scenes = WSI.num_scenes
    SVS_scene_name = scene.name
    SVS_scene_rect = scene.rect
    SVS_scene_num_channels = scene.num_channels
    SVS_scene_resolution = scene.resolution
        
    
    if xmlexist:
        # # # # # # # # # # # # # # # # # # # Read XML file
        xmlpath = str.replace(file, 'svs', 'xml')
        print('ReadingXML: ' + xmlpath)
        #opening the xml file in read mode
        with open(xmlpath,"r") as xml_obj:
            #coverting the xml data to Python dictionary
            my_dict = xmltodict.parse(xml_obj.read())
            #closing the file
            xml_obj.close()
        
        
        
        # # # # # # # # # # # # # # # # # # # Extraxt Regions
        xml = minidom.parse(xmlpath)
        # The first region marked is always the tumour delineation
        regions_ = xml.getElementsByTagName("Region")
        regions, region_labels = [], []
        XML_num_CAH = 0
        XML_num_Car = 0
        XML_num_Ben = 0
        CAH_Area = 0
        Carcinoma_Area = 0
        Benign_Area = 0
        CAH_Area_Micron = 0
        Carcinoma_Area_Micron = 0
        Benign_Area_Micron = 0
        
        CAH_Length = 0
        Carcinoma_Length = 0
        Benign_Length = 0
        CAH_Length_Micron = 0
        Carcinoma_Length_Micron = 0
        Benign_Length_Micron = 0
        
        for region in regions_:
            vertices = region.getElementsByTagName("Vertex")
            attribute = region.getElementsByTagName("Attribute")

            r_label = region.parentNode.parentNode.getAttribute('Name')
            if r_label == 'CAH':
                XML_num_CAH = XML_num_CAH + 1
                CAH_Area = CAH_Area+float(region.getAttribute("Area"))
                CAH_Area_Micron = CAH_Area_Micron+float(region.getAttribute("AreaMicrons"))
                CAH_Length = CAH_Length+float(region.getAttribute("Length"))
                CAH_Length_Micron = CAH_Length_Micron+float(region.getAttribute("LengthMicrons"))
            elif r_label == 'Carcinoma':
                XML_num_Car = XML_num_Car + 1
                Carcinoma_Area = Carcinoma_Area+float(region.getAttribute("Area"))
                Carcinoma_Area_Micron = Carcinoma_Area_Micron+float(region.getAttribute("AreaMicrons"))
                Carcinoma_Length = Carcinoma_Length+float(region.getAttribute("Length"))
                Carcinoma_Length_Micron = Carcinoma_Length_Micron+float(region.getAttribute("LengthMicrons"))
            elif r_label == 'Benign':
                XML_num_Ben = XML_num_Ben + 1
                Benign_Area = Benign_Area+float(region.getAttribute("Area"))
                Benign_Area_Micron = Benign_Area_Micron+float(region.getAttribute("AreaMicrons"))
                Benign_Length = Benign_Length+float(region.getAttribute("Length"))
                Benign_Length_Micron = Benign_Length_Micron+float(region.getAttribute("LengthMicrons"))
                
            region_labels.append(r_label) 
        
        XML_num_Regions = len(region_labels)
        print(str(XML_num_Regions)+' Regions, CAH:'+str(XML_num_CAH)+', Carcinoma:'+str(XML_num_Car)+', Benign:'+str(XML_num_Ben))
        XML_uniqueLabels = set(region_labels)
        print(XML_uniqueLabels)   

        
    else:
        print("XML File Missing")
        CAH_Area = 0
        Carcinoma_Area = 0
        Benign_Area = 0
        XML_num_CAH = 0
        XML_num_Car = 0
        XML_num_Ben = 0
        XML_num_Regions = 0
        XML_uniqueLabels = []
        CAH_Area_Micron = 0
        Carcinoma_Area_Micron = 0
        Benign_Area_Micron = 0
        CAH_Length = 0
        Carcinoma_Length = 0
        Benign_Length = 0
        CAH_Length_Micron = 0
        Carcinoma_Length_Micron = 0
        Benign_Length_Micron = 0
        
    
    tmp.append(
        {
            'Patient ID' : PatiendID,
            'SVS File Name' : file,  
            'Responder' : Responder,
            'Time Point' : timePoint,
            'File Exist In 3DHist': FileExistIn3DHist,
            'SVS File Size' : SVSFileSize,
            'SVS Magnification':SVS_Magnification,
            'SVS MPP': SVS_MPP,
            'XML File Exist': xmlexist,
            'XML Unique Labels' : XML_uniqueLabels,
            'XML Num Regions' : XML_num_Regions,
            'Num CAH': XML_num_CAH,
            'Num Carcinoma': XML_num_Car,
            'Num Benign': XML_num_Ben,
            'Total CAH Area' : CAH_Area,
            'Total Carcinoma Area' : Carcinoma_Area,
            'Total BenignArea' : Benign_Area, 
            'CAH_Area_Micron' : CAH_Area_Micron,
            'Carcinoma_Area_Micron' : Carcinoma_Area_Micron,
            'Benign_Area_Micron' : Benign_Area_Micron,
            'CAH_Length' : CAH_Length,
            'Carcinoma_Length' : Carcinoma_Length,
            'Benign_Length' : Benign_Length,
            'CAH_Length_Micron' : CAH_Length_Micron,
            'Carcinoma_Length_Micron' : Carcinoma_Length_Micron,
            'Benign_Length_Micron' : Benign_Length_Micron,
            'SVS Num Scenes' : SVS_num_scenes,
            'SVS Scene Name' : SVS_scene_name,
            'SVS Rect' : SVS_scene_rect,
            'SVS Num Channels' : SVS_scene_num_channels,
            'SVS Resolution' : SVS_scene_resolution,
            'SVS MetaData' : SVS_MetaData,
            
            }
        )
    
datasetInfo = pd.DataFrame(tmp)
datasetInfo.head() 

if os.name == 'nt':
    print("File Saved on Windows!")
    datasetInfo.to_csv('C:/DATA/' +datasetName+'_datasetInfo_'+ver+'.csv',index=False)
else:
    print("File Saved on Ubuntu!")  
    datasetInfo.to_csv('/home/seyedm.mousavikahaki/Documents/' +datasetName+'_datasetInfo_'+ver+'.csv',index=False)



### Plot Responders Count
data = datasetInfo.loc[datasetInfo['File Exist In 3DHist'] == True]
data = data.loc[datasetInfo['XML File Exist'] == True]
##### Responder
data["Responder"] = np.where(data["Responder"] == 'y', "Yes", data["Responder"])
data["Responder"] = np.where(data["Responder"] == 'n', "No", data["Responder"])
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






    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    