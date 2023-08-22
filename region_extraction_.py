# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Sep 12 12:36:48 2021
@author: Seyed
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
# import openslide
# from openslide import open_slide  
# from openslide.deepzoom import DeepZoomGenerator
# from glob import glob

ROOTPATH = "C:/DATA/"
DATAPATH = ''


# Returns a list of names in list files.
print("Using glob.glob() to read svs files")
files = glob.glob(ROOTPATH+DATAPATH+'*.svs', 
                   recursive = True)



# Read svs files
for file in files:
    print("--------------------------------")
    print('ReadingSVS: ' + file)
    
    WSI = slideio.open_slide(file, 'SVS') 
    num_scenes = WSI.num_scenes
    scene = WSI.get_scene(0)
    print(num_scenes,scene.name,scene.rect,scene.num_channels,scene.resolution)
    
    row_string = WSI.raw_metadata
    row_string.split("|")
    
    for channel in range(scene.num_channels):
        print(scene.get_channel_data_type(channel))

    image = scene.read_block(size=(500,0))
    plt.imshow(image)

    xmlpath = str.replace(file, 'svs', 'xml')
    print('ReadingXML: ' + xmlpath)
    #opening the xml file in read mode
    with open(xmlpath,"r") as xml_obj:
        #coverting the xml data to Python dictionary
        my_dict = xmltodict.parse(xml_obj.read())
        #closing the file
        xml_obj.close()
    
    print(my_dict)
    
    
    
    xml = minidom.parse(xmlpath)
    # The first region marked is always the tumour delineation
    regions_ = xml.getElementsByTagName("Region")
    regions, region_labels = [], []
    for region in regions_:
        vertices = region.getElementsByTagName("Vertex")
        attribute = region.getElementsByTagName("Attribute")
    #     if len(attribute) > 0:
    #         r_label = attribute[0].attributes['Value'].value
    #     else:
    #         r_label = region.getAttribute('Text')
        r_label = region.parentNode.parentNode.getAttribute('Name')
        region_labels.append(r_label)
        
        # Store x, y coordinates into a 2D array in format [x1, y1], [x2, y2], ...
        coords = np.zeros((len(vertices), 2))
        
        for i, vertex in enumerate(vertices):
            coords[i][0] = vertex.attributes['X'].value
            coords[i][1] = vertex.attributes['Y'].value
            
        regions.append(coords)



regions[0]



region_ims = []
for idx, region in enumerate(regions):
    print('_____________ Region:'+str(idx)+'______________')
    print(region_labels[idx])
    print(region)
    
    
    region = region.round().astype(int)
    xstart = region[0,0]
    ystart = region[0,1]
    xsize = max(region[:,0])-min(region[:,0])
    ysize = max(region[:,1])-min(region[:,1])
    
    region_im = scene.read_block((xstart,ystart,xsize,ysize))
    plt.imshow(region_im) 
    plt.title(region_labels[idx])
    plt.show()
    
    region_ims.append(region_im)

len(region_im)
region_im[0].shape

region



