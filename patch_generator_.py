'''
#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# ---------------------------------------------------------------------------
# Created on Mon Nov  6 11:23:07 2021
#
# @author: SeyedM.MousaviKahaki (mousavikahaki@gmail.com)
#----------------------------------------------------------------------------
# Title:        WSI Patch Generatore
#
# Description:  This code inputs should be run after the TissueRefinement code
#               It will extract several patches ($Number_of_Patches=300)
#               from specified images
#
#
# Input:        Image: Extracted Annotation
# Output:       Several patches with specific size
#
#
# Example:      patch_generator_.py --source DATA_DIRECTORY --save_dir RESULTS_DIRECTORY --dataset_file DATASET_FILE
#               OR
#               runfile('patch_generator_.py', args='--source "C:/DATA/2_extracted_cutted_Augmented/3DHistech/ExtractedAnnotation/" --save_dir "C:/DATA/2_extracted_cutted_Augmented/3DHistech/" --dataset_file "C:/DATA/Aperio_dataset_v9.csv"')
#
#
# version ='3.0'
# ---------------------------------------------------------------------------
'''
##############################   General Imports
import random
import cv2
import numpy as np
import os
from pathlib import Path
from PIL import Image
import glob
import h5py
from skimage.io import imsave, imread
from skimage.transform import resize
from pathlib import Path
import sys
import pandas as pd
from scipy import misc
import matplotlib.pyplot as plt

from skimage.metrics import structural_similarity as compare_ssim
import argparse
##############################   Internal Imports
# sys.path.insert(1, 'C:/DATA/Code/DigiPath_OWH/Config')
import parameters




# INPUTDIR = 'Oklahoma_Extracted_New_FixedCircle_processed_Augmented_combined/'
# OUTPUTDIR = 'C:/DATA/Oklahoma_extracted_cutted_Augmented/'
# DatasetFile = 'C:/DATA/Aperio_dataset_v9.csv'

# INPUTDIR = 'AperioAnnotations_NonCancer/'
# OUTPUTDIR = 'C:/DATA/AperioAnnotations_NonCancer_Patches/'
# DatasetFile = 'C:/DATA/Aperio_dataset_v9.csv'

# INPUTDIR = 'C:/DATA/2_extracted_cutted_Augmented/3DHistech/ExtractedAnnotation/'
# OUTPUTDIR = 'C:/DATA/2_extracted_cutted_Augmented/3DHistech/'
# DatasetFile = 'C:/DATA/Aperio_dataset_v9.csv'


parser = argparse.ArgumentParser(description='annotation extraction')

parser.add_argument('--source', type = str,
                    help='path to folder containing annotation files extracted from WSIs')
parser.add_argument('--save_dir', type = str,
                    help='path to folder to save extracted patches')
parser.add_argument('--dataset_file', type = str,
                    help='path to the dataset file')
parser.add_argument('--verbose', action='store_true', default=False)

intensity_check_thr = 230
meanintensity_check_thr = 250

#for on all folders
def main():
    args = parser.parse_args()

    INPUTDIR = args.source
    OUTPUTDIR= args.save_dir
    DatasetFile = args.dataset_file
    # cwd = os.getcwd()
    # # Create folders if there aren't
    # hdf5_dir = Path("data\\hdf5_files\\")
    # hdf5_dir.mkdir(parents=True, exist_ok=True)
    # png_dir = Path("data/png_files/")
    # hdf5_dir.mkdir(parents=True, exist_ok=True)
    hdf5_file= OUTPUTDIR+"dataset.hdf5"
    root_directory = glob.glob(r''+INPUTDIR+'*')

    if args.verbose:
        print(root_directory)

    png_dir = Path(OUTPUTDIR+"data/png_files/")
    png_dir.mkdir(parents=True, exist_ok=True)

    # path to str -We need them ahead
    # png_dir=r"data/png_files/"
    #hdf5_dir=r"data\\hdf5_files\\"
    png_dir=OUTPUTDIR+"data/png_files/"


    Dataset_ = pd.read_csv(DatasetFile)

    ##############################  inputs
    png_extract = parameters.PatchGenerator_png_extract
    hdf5 = False #parameters.PatchGenerator_hdf5
    open_dataset= parameters.PatchGenerator_open_dataset
    Aplha = parameters.PatchGenerator_Aplha
    # INPUTDIR = parameters.PatchGenerator_INPUTDIR
    # OUTPUTDIR = parameters.PatchGenerator_OUTPUTDIR
    ##############################  Number of Random Patches Per Region
    Number_of_Patches = parameters.PatchGenerator_Number_of_Patches
    ##############################  input patch size  y==width   x==height
    input_x = parameters.PatchGenerator_input_x
    input_y = parameters.PatchGenerator_input_y

    ##############################




    for filename in root_directory:

        #groupname = filename.split("\\")[-1]
        #groupname = filename.split("/")[-1]
        groupname = os.path.basename(filename)
        if args.verbose:
            print(groupname)

        # Find Magnification
        FName = groupname.upper() + '.SVS'
        print("Processing " + FName)

        try:
            # For Aperio
            Magnification = Dataset_[Dataset_['Filename of initial Aperio slide'] == FName]['SVS Magnification'].item()
            # For 3DHistech
            #Magnification = 40#Dataset_[Dataset_['Filename of initial 3D Histech slide'] == FName]['SVS Magnification'].item()

            if args.verbose:
                print(Magnification)
            if Magnification == 40:
                if args.verbose:
                    print(groupname+' Magnification  is 40')
            elif Magnification == 13:
                if args.verbose:
                    print(groupname+' Magnification  is 13')
            else:
                if args.verbose:
                    print('Magnification is 20')

            chck_group_name=True

            #files = glob.glob(filename + r"/*.jpg")
            files = os.listdir(filename)
            files = [os.path.join(filename, f) for f in files]

            files_clean = []
            # exclude mask images
            # i=0
            for r in files:
                a = os.path.basename(r)
                b= a.split(".")[0]
                c=b.find("mask")
                # if(c>-1):
                #     rt=files.pop()[i]\
                if(c==-1):
                    files_clean.append(r)
                # i=i+1


            # for on  all images in a folder
            for fl in files_clean:

                img = cv2.imread(fl, cv2.IMREAD_COLOR)

                if Magnification == 13:
                    if args.verbose:
                        print(groupname+' Rescaling 13X to 20X')
                    scale = 0.65
                    img = cv2.resize(img, (0,0), fx=scale, fy=scale, interpolation=cv2.INTER_AREA)

                elif Magnification == 40:
                    if args.verbose:
                        print(groupname+' Rescaling 40X to 20X')
                    scale = 0.5
                    img = cv2.resize(img, (0,0), fx=scale, fy=scale, interpolation=cv2.INTER_AREA)


                else:
                    if args.verbose:
                        print('Processing 20X')

                # Don't use regions that are too small
                print(img.shape)
                min_acceptable_height, min_acceptable_width = 2*input_y, 2*input_x
                if img.shape[0] < min_acceptable_height or img.shape[1] < min_acceptable_width:
                    continue

                if np.mean(img) > meanintensity_check_thr:
                    # TODO: describe what this is for
                    continue

                rows,cols = img.shape[0], img.shape[1]

                b = os.path.basename(fl)

                name=b.split('.')[0]

                #extract patches
                for rng in range(Number_of_Patches):

                    end_name =name+"_patchnumber_"+str(rng)

                    done = True
                    # note that breakLimit may need to be increased if not enough patches are being generated...
                    breakLimit = 500
                    breakCount = 0
                    savedPatchesCount = 0
                    while done:
                        if args.verbose:
                            print("Trying Extract: "+ name + "Break Count: " + str(breakCount))
                        breakCount = breakCount + 1
                        if breakCount > breakLimit:
                            print(">>>>>>>>>>> BREAK on "+ name)
                            print("breakCount = {}".format(breakCount))
                            print("savedPatchesCount = {}".format(savedPatchesCount))
                            break
                        coords = [(random.random()*rows, random.random()*cols)]
                        x=int(coords[0][0])
                        if(x>(rows-input_x+1)):
                            x = x-input_x+1

                        y=int(coords[0][1])
                        if(y>(cols-input_x+1)):
                            y = y-input_y+1

                        x_end=x+input_x
                        y_end=y+input_y

                        try:
                            # TODO: describe what this is for
                            color_chk1 = img[x, y] > [intensity_check_thr,intensity_check_thr,intensity_check_thr]
                            color_chk2 = img[x, y_end] > [intensity_check_thr,intensity_check_thr,intensity_check_thr]
                            color_chk3 = img[x_end, y] > [intensity_check_thr,intensity_check_thr,intensity_check_thr]
                            color_chk4 = img[x_end, y_end] > [intensity_check_thr,intensity_check_thr,intensity_check_thr]#== [255,255,255]
                            color_chk5 = img[round((x+x_end)/2), round((y+y_end)/2)] > [intensity_check_thr,intensity_check_thr,intensity_check_thr]
                        except Exception as e:
                            if args.verbose:
                                print(e)
                            continue
                        if any(color_chk1) == any(color_chk2) == any(color_chk3) == any(color_chk5) == False :
                            cropped_image = img[x:x_end, y:y_end]
                            if not cropped_image.shape == (input_y, input_x, 3):
                                print("patch dimensions are incorrect")
                                continue

                            #create png files of patches
                            if png_extract==True:
                                png_file = Path(OUTPUTDIR+"data/png_files/"+groupname+"/")
                                png_file.mkdir(parents=True, exist_ok=True)
                                if args.verbose:
                                    print("Crearing "+png_dir +groupname+"/"+end_name+".png")
                                try:
                                    cv2.imwrite(png_dir + groupname + "/" +end_name+ "_xy" + str(x) + "_" + str(y) + ".png", cropped_image)
                                    savedPatchesCount += 1
                                except Exception as e:
                                    if args.verbose:
                                        print("failed cv2.imwrite for:")
                                        print(png_dir + groupname + "/" +end_name+ "_xy" + str(x) + "_" + str(y))
                                        print(e)
                                    continue


                            #create a hdf5 file of all patches
                            if(hdf5==True):
                                if open_dataset==True:
                                    dataset = h5py.File(hdf5_file, 'a')
                                    open_dataset=False

                                if chck_group_name==True:

                                    if args.verbose:
                                        print(groupname+"_is-->>>>> new group name.")
                                    grp = dataset.create_group(groupname);
                                    chck_group_name=False



                                dset = grp.create_dataset(end_name, data=cropped_image)
                                if args.verbose:
                                    print(end_name+"_is new dataset on  "+groupname+" group")

                            done=False
        except:
            print("*"*100)
            print("Can not Process "+FName)
            print("*"*100)
    if(hdf5==True):
        dataset.close()

if __name__ == '__main__':
    main()

