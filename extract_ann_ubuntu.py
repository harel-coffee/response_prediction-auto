#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# ---------------------------------------------------------------------------
# Created on Wed Sep 29 10:35:19 2021
#
# @author: SeyedM.MousaviKahaki (mousavikahaki@gmail.com)
#----------------------------------------------------------------------------
# Title:        Extract Annotation (Ubuntu)
#
# Description:  This code extract different types of annotations including
#               Rectangle, Circle, Polygon, Ellipse, and Free form annotation
#               This code runs on Ubuntu system, for windows, use extract_ann_windows.py
#
# Input:        String: Source directory where the WSIs and corresponding XML files are located  
# Output:       Extracted annotations
#
# 
# Example:      extract_ann_ubuntu.py
#
#
# version ='3.0'
# ---------------------------------------------------------------------------

import cv2
import numpy as np
import os
import openslide
from openslide import open_slide  
import lxml.etree as ET
import lxml
from glob import glob
from skimage.io import imsave, imread
from PIL import Image
import math
#when svs is 40x highsize is True 
highsize =True          


save_dir = '/home/seyedm.mousavikahaki/Documents/wxc4/extracted'
WSIs_ = glob('/home/seyedm.mousavikahaki/Documents/wxc4/*.svs')
# WSIs_ = glob('/gpfs_projects/wxc4/DigiPath-WashU-Data/Washu-Aperio/*.svs')

WSIs = []
XMLs = []

for WSI in WSIs_:
    xml_ = str.replace(WSI, 'svs', 'xml')
    xmlexist = os.path.exists(xml_)
    if xmlexist:
        print('including: ' + WSI)
        XMLs.append(xml_)
        WSIs.append(WSI)

def main():
    # go though all WSI
    for idx, XML in enumerate(XMLs):
        annotationID=0
        tree = ET.parse(XML)
        root = tree.getroot()
        annot = root.findall("./Annotation")
        for annotation in annot:
            final_x=[]
            final_y=[] 
            annotationID=annotationID+1
            Regions = root.findall("./Annotation[@Id='" + str(annotationID) + "']/Regions/Region")
            bounds = []
            masks = []  
            lbl = annotation.attrib['Name']
            region_number=0
            for Region in Regions:  
                Vertices = Region.findall("./Vertices/Vertex")
                x = []
                y = []
                for Vertex in Vertices:
                    x.append(int(np.float32(Vertex.attrib['X'])))
                    y.append(int(np.float32(Vertex.attrib['Y'])))
                    
                x1=x  #new------------------------------------
                y1=y  ##new------------------------------------
                if highsize== True:#new------------------------------------
                    x = [ math.floor(number/4) for number in x]#new------------------------------------
                    y = [math.floor(number/4) for number in y]#new------------------------------------
                
                
                # x_center = min(x) + ((max(x)-min(x))/2)
                # y_center = min(y) + ((max(y)-min(y))/2)
                final_x.append(max(x) - min(x))
                final_y.append(max(y) - min(y))
                # bound_x = x_center - final_x[region_number] /2
                # bound_y = y_center-final_y[region_number]/2
                bounds.append([min(x1), min(y1)])#new------------------------------------
                
                # bounds.append([min(x), min(y)])
                points = np.stack([np.asarray(x), np.asarray(y)], axis=1)
                points[:,1] = np.int32(np.round(points[:,1] - min(y) )) 
                points[:,0] = np.int32(np.round(points[:,0] - min(x) ))
                mask = np.zeros([final_y[region_number], final_x[region_number]], dtype=np.int8)
                
                
                
                # cv2.fillPoly(mask, [points], color=(255,255,255))
                if(len(x)>3):                                        #polygon
                    print("poly")
                    cv2.fillPoly(mask, [points], color=(255,255,255))
                
                if(len(x)==2):                            # cirlce or ellipse
                    
                    x_c_int = int(points[1,0]/2)
                    y_c_int =int(points[1,1]/2)
                    center_coordinates=(x_c_int,y_c_int)
                    
                    if(points[:,0][1]==points[:,1][1]):        #cirlce
                        print("circle ")
                        cv2.circle(img=mask, center =center_coordinates, radius=x_c_int , color =(255,255,255), thickness=-1)
                        
                    else:                                       # ellipse
                        print("ellipse ")
                        
                        axesLength = (x_c_int, y_c_int)
                        angle =0
                        startAngle = 0
                        endAngle = 360
                        image = cv2.ellipse(mask, center_coordinates, axesLength, angle, startAngle, endAngle, (255,255,255), thickness=-1)
               ##### <-----new block
                
                
                
                basename = os.path.basename(XML)
                basename = os.path.splitext(basename)[0]
                subdirm = '{}/{}/'.format(save_dir,basename)
                # cv2.imwrite(subdirm+basename+"_anno_"+str(annotationID)+"_reg_"+str(region_number+1)+"_mask_"+lbl+".jpg",mask)
                print('saved <<<mask>>> of annotationID : ' + str(annotationID))
                print(subdirm+basename+"_anno_"+str(annotationID)+"_reg_"+str(region_number+1)+"_mask_"+lbl+".jpg")
                masks.append(mask)
                print('opening: a region of' + WSIs[idx])
                pas_img = openslide.OpenSlide(WSIs[idx])
                mask = masks[region_number]
                if highsize== True:         #new------------------------------------
                    PAS = pas_img.read_region((int(bounds[region_number][0]),  #new------------------------------------
                                               int(bounds[region_number][1])), #new------------------------------------
                                              1,(final_x[region_number],final_y[region_number]))#new------------------------------------
                    # plt.imshow(PAS)#new------------------------------------
                else:#new------------------------------------
                    PAS = pas_img.read_region((int(bounds[region_number][0]),#new------------------------------------
                                               int(bounds[region_number][1])),#new------------------------------------
                                              0, (final_x[region_number],final_y[region_number]))#new------------------------------------
                    # plt.imshow(PAS)#new------------------------------------
                
                # PAS = pas_img.read_region((int(bounds[region_number][0]),int(bounds[region_number][1])), 0, (final_x[region_number],final_y[region_number]))
                PAS = np.array(PAS)[:,:,0:3]
                for channel in range(3):
                    PAS_ = PAS[:,:,channel]
                    PAS_[masks[region_number] == 0] = 255
                    PAS[:,:,channel] = PAS_
                subdir = '{}/{}/'.format(save_dir,basename)
                make_folder(subdir)
                imsave(subdir + basename + '_anno_' + str(annotationID) +"_reg_"+str(region_number+1)+lbl+ '.jpg', PAS) 
                print('saved <region> of annotationID : ' + str(annotationID))
                region_number = region_number + 1
                
cv2.destroyAllWindows()  ##### new              

def make_folder(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)


if __name__ == '__main__':
    main()
