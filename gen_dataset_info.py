#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# ---------------------------------------------------------------------------
# Created on Sat Sep 18 07:48:39 2021
#
# @author: SeyedM.MousaviKahaki (mousavikahaki@gmail.com)
#----------------------------------------------------------------------------
# Title:        Generate Dataset File
#
# Description:  This code Read the WSI files and XML along with the patient data 
#               and Generates the final dataset to use
#
#
# Input:        String: ROOTPATH: Data Directory
#               String: Dataset Name 
# Output:       CSV file: training and test subsets
#
# 
# Example:      gen_dataset.py
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

if os.name == 'nt':
    print("Running on Windows!")
    ROOTPATH = 'C:/DATA/'
else:
    print("Running on Ubuntu!")  
    ROOTPATH = '/gpfs_projects/wxc4/DigiPath-WashU-Data/'    

datasetName = 'Aperio_Oklahoma' # 'Aperio' '3Dhistech' 'Aperio_Oklahoma' '3DHistech_Oklahoma'
 
ver = 'v_oklahoma'
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


FileExistIn3DHist = 0
misscount = 0
tmp = []
# Read svs files
for file in files:
    print("--------------------------------")
    print('ReadingSVS: ' + file)
    
    if datasetName == 'Aperio':
        if file.find('phantom') != -1:
            aperioFileNum = []
        elif file.find('new') != -1: 
            strFilenum = file[len(ROOTPATH+DATAPATH+'aperio-'):-len('new-0.svs')]
            aperioFileNum = int(strFilenum)
        else:
            strFilenum = file[len(ROOTPATH+DATAPATH+'aperio-'):-len('-0.svs')]
            aperioFileNum = int(strFilenum)
        
        print(aperioFileNum)
        timePoint = int(file[len(ROOTPATH+DATAPATH+'aperio-'+strFilenum+'-'):-len('.svs')])
        
        path3Dhist = ROOTPATH + 'WashU-3DHistech/' + '3Dhistech-'+str(aperioFileNum)+'-0.svs'
        path3Dhistnew = ROOTPATH + 'WashU-3DHistech/' + '3Dhistech-'+str(aperioFileNum)+'new-0.svs'
        
        FileExistIn3DHist = (os.path.exists(path3Dhist)|os.path.exists(path3Dhistnew))
        print(FileExistIn3DHist)
        if FileExistIn3DHist != True:
            misscount = misscount +1
    else:
        FileExistIn3DHist = 0
    

    SVSFileSize = round(os.path.getsize(file) / (1024*1024*1024),2)
    xmlpath = str.replace(file, 'svs', 'xml')
    xmlexist = os.path.exists(xmlpath)
    
    
    # # # # # # # # # # # # # # # # # # # Read SVS file
    WSI = slideio.open_slide(file, 'SVS') 
    scene = WSI.get_scene(0)
    row_string = WSI.raw_metadata
    
    SVS_MetaData = row_string.split("|")
    if datasetName == 'Aperio':
        if aperioFileNum == 27:
            print(SVS_MetaData)
    
    # Get APPMag from MetaData
    SVS_Magnification = []
    given_string = row_string
    start_string = 'AppMag = '
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
        #     if len(attribute) > 0:
        #         r_label = attribute[0].attributes['Value'].value
        #     else:
        #         r_label = region.getAttribute('Text')
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
        
        
        # TODO 
        # Check size of regions (later)
        
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
    
    tmp.append(
        {
            'Patient ID' : aperioFileNum,
            'SVS File Name' : file,            
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
            # 'Magnification': SVS_MetaData[1],
            # 'MPP': SVS_MetaData[10],
            # 'OriginalWidth': SVS_MetaData[23],
            # 'OriginalHeight': SVS_MetaData[24],
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



    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    