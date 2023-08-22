# -*- coding: utf-8 -*-
"""
Created on Tue Nov  6 10:23:07 2021

@author: SeyedM.MousaviKahaki
"""
import random
import cv2
# import numpy as np
# import os
from pathlib import Path
# from PIL import Image
import glob
import h5py
# from skimage.io import imsave, imread
# from skimage.transform import resize
# from pathlib import Path


##############################
#inputs
png_extract = True
hdf5 = True
Number_of_Patches = 10

#input size  y==width   x==height
input_x = 200    
input_y = 400
##############################
# cwd = os.getcwd()

INPUTDIR = 'C:/DATA/extracted/'
OUTPUTDIR = 'C:/DATA/extracted_cutted/'

# input polygons address (defult are jpg files)
# files = glob.glob( cwd + r"\data\polygons\*.jpg")
files = glob.glob(INPUTDIR+"aperio-002-0/*.jpg")
print(files)


# Create folders if there aren't
hdf5_dir = Path(OUTPUTDIR+"data\\hdf5_files\\")
hdf5_dir.mkdir(parents=True, exist_ok=True)
png_dir = Path(OUTPUTDIR+"data/png_files/")
png_dir.mkdir(parents=True, exist_ok=True)

# path to str -We need them ahead
png_dir=OUTPUTDIR+"data/png_files/"
hdf5_dir=OUTPUTDIR+"data/hdf5_files/"


#for on polygons
for filename in files:
    meta = []
    print("--------------***------------------")
    img = cv2.imread(filename, cv2.IMREAD_COLOR)
    
    rows,cols = img.shape[0], img.shape[1]
    
    print(filename)
    
    b=filename.split('\\')[-1]
    name=b.split('.')[0]

    # cutting patches 
    for rng in range(Number_of_Patches):
        name_end = name+str(rng)    
        done = True
        while done : 
            coords = [(random.random()*rows, random.random()*cols)]
            x=int(coords[0][0])
            if(x>(rows-input_x+1)):
                x = x-input_x+1
            #print(x , rows)
            y=int(coords[0][1])
            if(y>(cols-input_x+1)):
                y = y-input_y+1
            #print(y ,cols)
            x_end=x+input_x
            y_end=y+input_y
            #print(x_end , y_end)  
            try:
                color_chk1 = img[x, y] == [255,255,255]
                color_chk2 = img[x, y_end] == [255,255,255]
                color_chk3 = img[x_end, y] == [255,255,255]
                color_chk4 = img[x_end, y_end] == [255,255,255]
            except:
                continue
            if any(color_chk1) == any(color_chk2) == any(color_chk3) == any(color_chk4) == False : 
                cropped_image = img[x:x_end, y:y_end]
                if png_extract==True:
                    cv2.imwrite(png_dir +name+'__'+str(rng)+".png", cropped_image)
                    print(png_dir +name+'__'+str(rng)+".png")
                if(hdf5==True):
                    hf1 = h5py.File(hdf5_dir+name+'.hdf5', 'a')
                    try:
                        dset1 = hf1.create_dataset(name_end, data=cropped_image)
                        print("                                  "+name_end+"   saved in  "+name+'.hdf5')
                    except:
                        print(">>>>>>>>  warning !!!!  Unable to create dataset (name already exists)")
                    hf1.close()
                done=False 

