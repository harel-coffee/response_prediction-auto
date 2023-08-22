#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# ---------------------------------------------------------------------------
# Created on Wed Jul  6 13:10:40 2022
#
# @author: SeyedM.MousaviKahaki (mousavikahaki@gmail.com)
#----------------------------------------------------------------------------
# Title:        Image Normalization
#
# Description:  This code normalize image patches 
#
#
# Input:        String: Source directory where the patches are located  
# Output:       Normalized patches
#
# 
# Example:      patch_normalization.py --source DATA_DIRECTORY --save_dir RESULTS_DIRECTORY --method NORMALIZATION_METHOD --target_img TARGET_IMAGE_PATH
#               OR
#               runfile('patch_normalization.py', args='--source "C:/DATA/2_extracted_cutted_Augmented/data/png_files/TempForNormalization" --save_dir "C:/DATA/2_extracted_cutted_Augmented/NormalizationResults/HandE_Norm/Aperio/" --method "H_E" --target_img = "C:/DATA/2_extracted_cutted_Augmented/data/png_files/TempForNormalization/aperio-001-0/aperio-001-0_anno_2_reg_3CAH_patchnumber_13.png"')
#
# version ='3.0'
# ---------------------------------------------------------------------------
"""
Created on Wed Jul  6 13:10:40 2022

@author: SeyedM.MousaviKahaki
"""

# for loading/processing the images  
from keras.preprocessing.image import load_img 
from keras.preprocessing.image import img_to_array 
from keras.applications.vgg16 import preprocess_input 
# load json module
import json
# models 
from keras.applications.vgg16 import VGG16 
from keras.models import Model
import cv2
# clustering and dimension reduction
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

# for everything else
import os
import numpy as np
import matplotlib.pyplot as plt
from random import randint
import pandas as pd
import pickle
from glob import glob
from skimage import exposure
from skimage.exposure import match_histograms  
import argparse
# WSIs_ = glob('C:/DATA/2_extracted_cutted_Augmented/data/png_files/TempForNormalization/*')
# OUTPUTDIR = 'C:/DATA/2_extracted_cutted_Augmented/NormalizationResults/HandE_Norm/Aperio/'


parser = argparse.ArgumentParser(description='patch normalization')

parser.add_argument('--source', type = str,
					help='path to folder containing images')
parser.add_argument('--save_dir', type = str,
					help='path to folder to save normalized images')

parser.add_argument('--method', type = str,
					help='image normalization method: H_E, Hist_match')

# parser.add_argument('--target_img', type = str,
# 					help='path to the target image. only used for histogram matching method')


def norm_HnE(img, Io=240, alpha=1, beta=0.15):

    # Workflow based on the following papers:
    # A method for normalizing histology slides for quantitative analysis. 
    # M. Macenko et al., ISBI 2009
    #     http://wwwx.cs.unc.edu/~mn/sites/default/files/macenko2009.pdf
    
    # Efficient nucleus detector in histopathology images. J.P. Vink et al., J Microscopy, 2013
    
    # Original MATLAB code:
    #     https://github.com/mitkovetta/staining-normalization/blob/master/normalizeStaining.m
     
    # Other useful references:
    #     https://www.ncbi.nlm.nih.gov/pmc/articles/PMC5226799/
    #     https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0169875
    # 
    # Python code based on https://youtu.be/tNfcvgPKgyU

    ######## Step 1: Convert RGB to OD ###################
    ## reference H&E OD matrix.
    #Can be updated if you know the best values for your image. 
    #Otherwise use the following default values. 
    #Read the above referenced papers on this topic. 
    HERef = np.array([[0.5626, 0.2159],
                      [0.7201, 0.8012],
                      [0.4062, 0.5581]])
    ### reference maximum stain concentrations for H&E
    maxCRef = np.array([1.9705, 1.0308])
    
    
    # extract the height, width and num of channels of image
    h, w, c = img.shape
    
    # reshape image to multiple rows and 3 columns.
    #Num of rows depends on the image size (wxh)
    img = img.reshape((-1,3))
    
    # calculate optical density
    # OD = −log10(I)  
    #OD = -np.log10(img+0.004)  #Use this when reading images with skimage
    #Adding 0.004 just to avoid log of zero. 
    
    OD = -np.log10((img.astype(np.float)+1)/Io) #Use this for opencv imread
    #Add 1 in case any pixels in the image have a value of 0 (log 0 is indeterminate)
    
    
    ############ Step 2: Remove data with OD intensity less than β ############
    # remove transparent pixels (clear region with no tissue)
    ODhat = OD[~np.any(OD < beta, axis=1)] #Returns an array where OD values are above beta
    #Check by printing ODhat.min()
    
    ############# Step 3: Calculate SVD on the OD tuples ######################
    #Estimate covariance matrix of ODhat (transposed)
    # and then compute eigen values & eigenvectors.
    eigvals, eigvecs = np.linalg.eigh(np.cov(ODhat.T))
    
    
    ######## Step 4: Create plane from the SVD directions with two largest values ######
    #project on the plane spanned by the eigenvectors corresponding to the two 
    # largest eigenvalues    
    That = ODhat.dot(eigvecs[:,1:3]) #Dot product
    
    ############### Step 5: Project data onto the plane, and normalize to unit length ###########
    ############## Step 6: Calculate angle of each point wrt the first SVD direction ########
    #find the min and max vectors and project back to OD space
    phi = np.arctan2(That[:,1],That[:,0])
    
    minPhi = np.percentile(phi, alpha)
    maxPhi = np.percentile(phi, 100-alpha)
    
    vMin = eigvecs[:,1:3].dot(np.array([(np.cos(minPhi), np.sin(minPhi))]).T)
    vMax = eigvecs[:,1:3].dot(np.array([(np.cos(maxPhi), np.sin(maxPhi))]).T)
    
    
    # a heuristic to make the vector corresponding to hematoxylin first and the 
    # one corresponding to eosin second
    if vMin[0] > vMax[0]:    
        HE = np.array((vMin[:,0], vMax[:,0])).T
        
    else:
        HE = np.array((vMax[:,0], vMin[:,0])).T
    
    
    # rows correspond to channels (RGB), columns to OD values
    Y = np.reshape(OD, (-1, 3)).T
    
    # determine concentrations of the individual stains
    C = np.linalg.lstsq(HE,Y, rcond=None)[0]
    
    # normalize stain concentrations
    maxC = np.array([np.percentile(C[0,:], 99), np.percentile(C[1,:],99)])
    tmp = np.divide(maxC,maxCRef)
    C2 = np.divide(C,tmp[:, np.newaxis])
    
    ###### Step 8: Convert extreme values back to OD space
    # recreate the normalized image using reference mixing matrix 
    
    Inorm = np.multiply(Io, np.exp(-HERef.dot(C2)))
    Inorm[Inorm>255] = 254
    Inorm = np.reshape(Inorm.T, (h, w, 3)).astype(np.uint8)  
    
    # Separating H and E components
    
    H = np.multiply(Io, np.exp(np.expand_dims(-HERef[:,0], axis=1).dot(np.expand_dims(C2[0,:], axis=0))))
    H[H>255] = 254
    H = np.reshape(H.T, (h, w, 3)).astype(np.uint8)
    
    E = np.multiply(Io, np.exp(np.expand_dims(-HERef[:,1], axis=1).dot(np.expand_dims(C2[1,:], axis=0))))
    E[E>255] = 254
    E = np.reshape(E.T, (h, w, 3)).astype(np.uint8)
    
    return (Inorm, H, E)




########################################## H&E Norm
########## Aperio

def main():
    
    args = parser.parse_args()
    source = args.source
    OUTPUTDIR = args.save_dir
    method = args.method
    print('here')
    # print(args.target_img)
    # target_img = args.target_img
    target_img = 'C:/DATA/temp/target.png'
    Target = cv2.imread(target_img, 1)
    WSIs_ = glob(source+'/*')
    print(source)
    print(WSIs_)
    for W in WSIs_:
        WSI = W.rsplit('\\', 1)[1]
        print(WSI)
    
        # WSI = 'BA-10-84 HE'
        path = r"C:/DATA/2_extracted_cutted_Augmented/data/png_files/TempForNormalization/"+WSI+"/"
        # change the working directory to the path where the images are located
        os.chdir(path)
        
        # this list holds all the image filename
        flowers = []
        saveDir = OUTPUTDIR + WSI
        if not os.path.exists(saveDir):
            os.mkdir(saveDir)
        # creates a ScandirIterator aliased as files
        with os.scandir(path) as files:
          # loops through each file in the directory
            for file in files:
                if file.name.endswith('.png'):
                    flowers.append(file.name)
                    img=cv2.imread(path+file.name, 1)
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    # plt.imshow(img)
                    if method == 'H_E':
                        norm_img, H_img, E_img = norm_HnE(img, Io=240, alpha=1, beta=0.15)
                        plt.imsave(saveDir+"/"+file.name, norm_img)
                    elif method == 'Hist_match':
                        reference = np.array(Target, dtype=np.uint8)
                        image = np.array(img, dtype=np.uint8)    
                        matched = match_histograms(image, reference,multichannel=True)
                        plt.imsave(saveDir+"/"+file.name, matched)
                    else:
                        print('Not Implementer Error')
                        
                    # plt.imshow(E_img)
                  # adds only the image files to the flowers list
                    # flowers.append(file.name)
if __name__ == '__main__':
    main()	               
# ########## 3DHistech

# WSIs_ = glob('C:/DATA/2_extracted_cutted_Augmented/3DHistech/data/png_files/*')
# OUTPUTDIR = 'C:/DATA/2_extracted_cutted_Augmented/NormalizationResults/HandE_Norm/3DHistech/'

# for W in WSIs_:
#     WSI = W.rsplit('\\', 1)[1]
#     print(WSI)

#     # WSI = 'BA-10-84 HE'
#     path = r"C:/DATA/2_extracted_cutted_Augmented/3DHistech/data/png_files/"+WSI+"/"

#     # change the working directory to the path where the images are located
#     os.chdir(path)
    
#     # this list holds all the image filename
#     flowers = []
#     saveDir = OUTPUTDIR + WSI
#     if not os.path.exists(saveDir):
#         os.mkdir(saveDir)
#     # creates a ScandirIterator aliased as files
#     with os.scandir(path) as files:
#       # loops through each file in the directory
#         for file in files:
#             if file.name.endswith('.png'):
#                 flowers.append(file.name)
#                 img=cv2.imread(path+file.name, 1)
#                 img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
#                 # plt.imshow(img)
#                 norm_img, H_img, E_img = norm_HnE(img, Io=240, alpha=1, beta=0.15)
#                 plt.imsave(saveDir+"/"+file.name, norm_img)
#                 # plt.imshow(E_img)
#               # adds only the image files to the flowers list
#                 # flowers.append(file.name)

# ########################################## ITK Normalization       
# ########## 3DHistech
# import itk

# from matplotlib import colors
# from matplotlib.ticker import PercentFormatter

# WSIs_ = glob('C:/DATA/2_extracted_cutted_Augmented/3DHistech/data/png_files/*')
# OUTPUTDIR = 'C:/DATA/2_extracted_cutted_Augmented/NormalizationResults/ITK/3DHistech/'

# for W in WSIs_:
#     WSI = W.rsplit('\\', 1)[1]
#     print(WSI)

#     # WSI = 'BA-10-84 HE'
#     path = r"C:/DATA/2_extracted_cutted_Augmented/3DHistech/data/png_files/"+WSI+"/"

#     # change the working directory to the path where the images are located
#     os.chdir(path)
    
#     # this list holds all the image filename
#     flowers = []
#     saveDir = OUTPUTDIR + WSI
#     if not os.path.exists(saveDir):
#         os.mkdir(saveDir)
#     # creates a ScandirIterator aliased as files
#     with os.scandir(path) as files:
#       # loops through each file in the directory
#         for file in files:
#             if file.name.endswith('.png'):
#                 flowers.append(file.name)



# #https://stackoverflow.com/questions/70233645/color-correction-using-opencv-and-color-cards

# ##USIN Panton
# # https://github.com/dazzafact/image_color_correction
# import matplotlib.pyplot as plt
# import numpy as np
# from skimage import data
# from skimage import exposure
# from skimage.exposure import match_histograms            


# Target0 = "C:/DATA/2_extracted_cutted_Augmented/data/png_files/TempForNormalization/aperio-001-0/aperio-001-0_anno_2_reg_3CAH_patchnumber_13.png"
# path = "C:/DATA/2_extracted_cutted_Augmented/3DHistech/data/png_files/3Dhistech-1-0/" 
# fileName = "3Dhistech-1-0_anno_2_reg_3CAH_patchnumber_23.png"#flowers[0]
# Target = img=cv2.imread(Target0, 1)
# plt.imshow(Target)
# Source = img=cv2.imread(path+fileName, 1)
# plt.imshow(Source)



# reference = np.array(Target, dtype=np.uint8)
# image = np.array(Source, dtype=np.uint8)


# matched = match_histograms(image, reference,multichannel=True)

# fig, (ax1, ax2, ax3) = plt.subplots(nrows=1, ncols=3, figsize=(8, 3), sharex=True, sharey=True)
# for aa in (ax1, ax2, ax3):
#     aa.set_axis_off()

# ax1.imshow(Source)
# ax1.set_title('Source')
# ax2.imshow(Target)
# ax2.set_title('Reference')
# ax3.imshow(matched)
# ax3.set_title('Matched')

# plt.tight_layout()
# plt.show()

# fig, axes = plt.subplots(nrows=3, ncols=3, figsize=(8, 8))


# for i, img in enumerate((Source, Target, matched)):
#     for c, c_color in enumerate(('red', 'green', 'blue')):
#         img_hist, bins = exposure.histogram(img[..., c], source_range='dtype')
#         axes[c, i].plot(bins, img_hist / img_hist.max())
#         img_cdf, bins = exposure.cumulative_distribution(img[..., c])
#         axes[c, i].plot(bins, img_cdf)
#         axes[c, 0].set_ylabel(c_color)

# axes[0, 0].set_title('Source')
# axes[0, 1].set_title('Reference')
# axes[0, 2].set_title('Matched')

# plt.tight_layout()
# plt.show()

# test = match_histograms(matched, image,multichannel=True)

# fig, (ax1, ax2, ax3) = plt.subplots(nrows=1, ncols=3, figsize=(8, 3), sharex=True, sharey=True)
# for aa in (ax1, ax2, ax3):
#     aa.set_axis_off()

# ax1.imshow(image)
# ax1.set_title('Source')
# ax2.imshow(matched)
# ax2.set_title('Reference')
# ax3.imshow(test)
# ax3.set_title('Matched')

# plt.tight_layout()
# plt.show()


# fig, axes = plt.subplots(nrows=3, ncols=3, figsize=(8, 8))


# for i, img in enumerate((image, matched, test)):
#     for c, c_color in enumerate(('red', 'green', 'blue')):
#         img_hist, bins = exposure.histogram(img[..., c], source_range='dtype')
#         axes[c, i].plot(bins, img_hist / img_hist.max())
#         img_cdf, bins = exposure.cumulative_distribution(img[..., c])
#         axes[c, i].plot(bins, img_cdf)
#         axes[c, 0].set_ylabel(c_color)

# axes[0, 0].set_title('Source')
# axes[0, 1].set_title('Reference')
# axes[0, 2].set_title('Matched')

# plt.tight_layout()
# plt.show()

                
                
                
                
                # input_image = itk.imread(path+file.name)
                # reference_image = itk.imread("C:/DATA/2_extracted_cutted_Augmented/data/png_files/TempForNormalization/aperio-001-0/aperio-001-0_anno_2_reg_3CAH_patchnumber_13.png")
                # eager_normalized_image = itk.StructurePreservingColorNormalizationFilter(
                # input_image,
                # reference_image,
                # color_index_suppressed_by_hematoxylin=0,
                # color_index_suppressed_by_eosin=1)
                # itk.imwrite(eager_normalized_image, saveDir+"/"+file.name)
                
             
# input_image_filename =     path+file.name
# reference_image_filename = "C:/DATA/2_extracted_cutted_Augmented/data/png_files/TempForNormalization/aperio-001-0/aperio-001-0_anno_2_reg_3CAH_patchnumber_13.png"

# # The pixels are RGB triplets of unsigned char.  The images are 2 dimensional.
# PixelType = itk.RGBPixel[itk.UC]
# ImageType = itk.Image[PixelType, 2]   
# # Alternatively, invoke the ITK pipeline
# input_reader = itk.ImageFileReader[ImageType].New(FileName=input_image_filename)
# reference_reader = itk.ImageFileReader[ImageType].New(FileName=reference_image_filename)


# itk.NormalizeToConstantImageFilter(
#                 input_image,
#                 reference_image,
#                 color_index_suppressed_by_hematoxylin=0,
#                 color_index_suppressed_by_eosin=1)


# spcn_filter = itk.StructurePreservingColorNormalizationFilter.New(Input=input_reader.GetOutput())
# spcn_filter.SetColorIndexSuppressedByHematoxylin(0)
# spcn_filter.SetColorIndexSuppressedByEosin(1)
# spcn_filter.SetInput(0, input_reader.GetOutput())
# spcn_filter.SetInput(1, reference_reader.GetOutput())
# output_writer = itk.ImageFileWriter.New(spcn_filter.GetOutput())
# output_writer.SetInput(spcn_filter.GetOutput())
# output_writer.SetFileName('')
# output_writer.Write()






















