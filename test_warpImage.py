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

def main(img_view1,img_view2,correspond_points_in_A,correspond_points_in_B):
    """
    Assumes a relative path of two views of the same image (including the .jpg expension) and the path of a numpy array with coordinates of corresponding points (inludes .npy)
    Returns a matplotlib figure that displays the image view B with the provided corresponding points and the same image with the corresponding points estimated
    using homography matrix 

    Note: This prototype main function is meant to run a rapid test of the warpImage function from question 3. Thereby, values are hardwired. 
    """

    # Loading images provided in assignment

    CROP1 = '..\\Zambrano_Ricardo_ASN4_py\\crop1.jpg'
    CROP2 = '..\\Zambrano_Ricardo_ASN4_py\\crop2.jpg'

    img_raw = io.imread(CROP1)
    crop1 = img_raw.copy()
    img_raw = io.imread(CROP2)
    crop2 = img_raw.copy()

    # Loading correspondent points provided in the assignment

    cc1 = np.load('..\\Zambrano_Ricardo_ASN4_py\\cc1.npy')
    cc2 = np.load('..\\Zambrano_Ricardo_ASN4_py\\cc2.npy')

    # Showing original image view A

    plt.figure()

    plt.imshow(crop1)
    plt.title("Original Input Image")

    plt.scatter(cc1[:,0],cc1[:,1])

    #plt.show() # Keep this line commented out to show both images at once

    # Showing original image view B

    plt.figure()

    plt.imshow(crop2)
    plt.title("Original Reference Image")

    plt.scatter(cc2[:,0],cc2[:,1])

    plt.show()

    print('Test start...')

    # Test begins by transposing points

    cc1_t = np.transpose(cc1)
    cc2_t = np.transpose(cc2)

    # Calculates H and collects minimum and maximum values of each image

    H, r1_min, r1_max, r2_min, r2_max = pre_process_correspondences(cc1_t,cc2_t)

    # Estimating warped image
    warped_img_view_B, mosaic_A_B = warpImage(crop1,crop2,H,r1_min,r1_max)

    # Final step displaying the original reference image and the inverse warp of said reference image

    fig = plt.figure()

    rows = 1
    columns = 3

    # Plot 1
    fig.add_subplot(rows, columns, 1) 
    
    # showing image 
    plt.imshow(crop2) 
    # plt.axis('off') 
    plt.title("Original Reference Image") 

    plt.scatter(cc2_t[0,:],cc2_t[1,:])
    
    # Plot 2
    fig.add_subplot(rows, columns, 2) 
    
    # showing image 
    plt.imshow(crop1) 
    # plt.axis('off') 
    plt.title("Original Input Image - Destination Frame") # <- Destination frame for reference because the requirement to do an inverse warp 

    plt.scatter(cc1_t[0,:],cc1_t[1,:])

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
    main(0,0,0,0)

