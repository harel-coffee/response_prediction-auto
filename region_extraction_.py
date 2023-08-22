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




# from shapely.geometry import Polygon, Point

# label_map = {'Carcinoma': 0,
#              'Benign': 1,
#              'CHA': 2,
#             }


# def generate_label(regions, region_labels, point):
#     # regions = array of vertices (all_coords)
#     # point [x, y]
#     for i in range(len(region_labels)):
#         poly = Polygon(regions[i])
#         if poly.contains(Point(point[0], point[1])):
#             return label_map[region_labels[i]]
#     return label_map['Normal']

# generate_label(regions, region_labels, [7500, 21600])

# patch_size = 256
# percent_overlap = 0
# file_dir = "wsi_data/"
# file_name = "A01.svs"
# xml_file = "A01.xml"
# xml_dir = "wsi_data/"
# level = 12

# overlap = int(patch_size*percent_overlap / 2.0)
# tile_size = patch_size - overlap*2

# slide = open_slide(file_dir + file_name) 
# tiles = DeepZoomGenerator(slide, tile_size=tile_size, overlap=overlap, limit_bounds=False)

# if level >= tiles.level_count:
#     print("Error: requested level does not exist. Slide level count: " + str(tiles.level_count))

# x_tiles, y_tiles = tiles.level_tiles[level]




# print(x_tiles)
# print(y_tiles)



# tiles.get_tile_coordinates(level, (5, 2))[0]

# patches, coords, labels = [], [], []
# x, y = 0, 0
# count = 0
# while y < y_tiles:
#     while x < x_tiles:
#         new_tile = np.array(tiles.get_tile(level, (x, y)), dtype=np.int)
#         # OpenSlide calculates overlap in such a way that sometimes depending on the dimensions, edge 
#         # patches are smaller than the others. We will ignore such patches.
#         if np.shape(new_tile) == (patch_size, patch_size, 3):
#             patches.append(new_tile)
#             coords.append(np.array([x, y]))
#             count += 1

#             # Calculate the patch label based on centre point.
#             if xml_file:
#                 converted_coords = tiles.get_tile_coordinates(level, (x, y))[0]
#                 labels.append(generate_label(regions, region_labels, converted_coords))
#         x += 1
#     y += 1
#     x = 0

# # image_ids = [im_id]*count

# print(np.shape(patches))
# print(np.shape(coords))
# print(np.shape(labels))

# np.count_nonzero(labels)

# name = 'tester.svs'

# name[:-4]









# # import multiresolutionimageinterface as mir
# # import os.path as osp
# # import glob
# # import re