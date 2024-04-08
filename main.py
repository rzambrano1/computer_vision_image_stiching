#!/usr/bin/python3

# Standard Libraries
import argparse
import os
import sys
from tqdm import tqdm

# Type Hint Libraries
from typing import Optional, Tuple, Union, TypeVar, List
import numpy.typing as npt
import matplotlib.figure

# Math Libraries
import numpy as np
from scipy.ndimage.filters import convolve

# Plot Libraries
import matplotlib.pyplot as plt
from matplotlib.backend_bases import MouseButton

# Machine Learning Libraries
from sklearn.cluster import KMeans
from sklearn.utils import shuffle

# Image Libraries
import cv2 

import skimage as ski
from skimage import io
from skimage.color import rgb2gray
from skimage.color import rgb2hsv
from skimage.color import hsv2rgb

# Functions Import
from getting_correspondences import getting_correspondences

from computeH import scale_down
from computeH import scale_back
from computeH import to_homogeneous
from computeH import from_homogeneous
from computeH import homogeneous_transform_coord
from computeH import computeH

from warpImage import pre_process_correspondences
from warpImage import calculate_inverse_frame
from warpImage import inverse_warping
from warpImage import warpImage

def main(input_image_name,reference_image_name):

    print("Loading images...")
    
    if input_image_name[-2:] == 'eg':
        PATH_IMG_A = '..\\Zambrano_Ricardo_ASN4_py\\' + input_image_name
        out_name_A = input_image_name[:-5]
    if input_image_name[-2:] == 'pg':
        PATH_IMG_A = '..\\Zambrano_Ricardo_ASN4_py\\' + input_image_name
        out_name_A = input_image_name[:-4]    

    if reference_image_name[-2:] == 'eg':
        PATH_IMG_B = '..\\Zambrano_Ricardo_ASN4_py\\' + reference_image_name
        out_name_B = reference_image_name[:-5]
    if reference_image_name[-2:] == 'pg':
        PATH_IMG_B = '..\\Zambrano_Ricardo_ASN4_py\\' + reference_image_name
        out_name_B = reference_image_name[:-4]

    PATH_IMG_A_PIXEL = '..\\Zambrano_Ricardo_ASN4_py\\' + out_name_A + '_pixel_coord' + '.npy'
    PATH_IMG_A_DATA = '..\\Zambrano_Ricardo_ASN4_py\\' + out_name_A + '_data_coord' + '.npy'

    PATH_IMG_B_PIXEL = '..\\Zambrano_Ricardo_ASN4_py\\' + out_name_B + '_pixel_coord' + '.npy'
    PATH_IMG_B_DATA = '..\\Zambrano_Ricardo_ASN4_py\\' + out_name_B + '_data_coord' + '.npy'

    img_raw_A = io.imread(PATH_IMG_A)
    img_raw_B = io.imread(PATH_IMG_B)

    img_A = img_raw_A.copy()
    img_B = img_raw_B.copy()

    print('Image sizes:')
    print('Image A ->',img_A.shape)
    print('Image B ->',img_B.shape)

    print("-------------------------")
    print("Two windows will pop up. Each window displays a view of the same image. Using your mouse click on pair of corresponding points in each image.")
    print("-------------------------")
    print("Corrsponding points are pixels that below to the same part of the same object in each different view.  The object may have change position or being viewed from a different angle. ")
    print("-------------------------")
    print("-------------------------")
    print("Windows might pop up on top of each other, move them appart by using your mouse - Select the top of the window only")
    print("Once all the similar points are clicked, right-click on each image to close them.")
    print("-------------------------")

    pixel_coord_A, pixel_coord_B, data_coord_A, data_coord_B = getting_correspondences(img_A,img_B)

    print("-------------------------\nCorrdinates in Image View A:\n")
    print(pixel_coord_A)
    print("-------------------------\nCorrdinates in Image View B:\n")
    print(pixel_coord_B)

    print('Saving coordinates...')
    np.save(PATH_IMG_A_PIXEL,pixel_coord_A)
    np.save(PATH_IMG_B_PIXEL,pixel_coord_B)
    np.save(PATH_IMG_A_DATA,data_coord_A)
    np.save(PATH_IMG_B_DATA,data_coord_B)

    # Next step: with the pixel correspondences calculating H and collect minimum and maximum values of each image

    print('Estimating homography matrix...')
    #H, r1_min, r1_max, r2_min, r2_max = pre_process_correspondences(pixel_coord_A,pixel_coord_B) <- Check
    H, r1_min, r1_max, r2_min, r2_max = pre_process_correspondences(pixel_coord_A,pixel_coord_B)

    # Estimating warped image
    print('Homography matrix completed...\nSampling from image B to get the warped image...')
    warped_img_view_B, mosaic_A_B = warpImage(img_A,img_B,H,r1_min,r1_max)

    # Final step displaying the original reference image and the inverse warp of said reference image

    fig = plt.figure()

    rows = 1
    columns = 3

    # Plot 1
    fig.add_subplot(rows, columns, 1) 
    
    # showing image 
    plt.imshow(img_B) 
    # plt.axis('off') 
    plt.title("Original Reference Image") 

    #plt.scatter(pixel_coord_B[0,:],pixel_coord_B[1,:])  <- Check
    plt.scatter(data_coord_B[0,:],data_coord_B[1,:])
    
    # Plot 2
    fig.add_subplot(rows, columns, 2) 
    
    # showing image 
    plt.imshow(img_A) 
    # plt.axis('off') 
    plt.title("Original Input Image - Destination Frame") # <- Destination frame for reference because the requirement to do an inverse warp 

    #plt.scatter(pixel_coord_A[0,:],pixel_coord_A[1,:])  <- Check
    plt.scatter(data_coord_A[0,:],data_coord_A[1,:])

    # Plot 3
    fig.add_subplot(rows, columns, 3) 
    
    # showing image 
    plt.imshow(warped_img_view_B)
    plt.title("Warped Image")

    plt.show()

    # Displaying the merged image

    plt.imshow(mosaic_A_B)
    plt.title("Merged Original Input and Warped Images")
    plt.show()

    return 0

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Pass two strings with the names of the JPG filea in the folder. The images are assigned to img_A and img_B in the scope of the function. img_B will be warped to img_A frame of reference.')

    parser.add_argument('-in','--input_image_name', type=str, default=False, action='store', required=True, help="String with name of input image, include file extension")
    parser.add_argument('-ref','--reference_image_name', type=str, default=False, action='store', required=True, help="String with name of reference image, include file extension")

    args = parser.parse_args()
    main(str(args.input_image_name), str(args.reference_image_name))
