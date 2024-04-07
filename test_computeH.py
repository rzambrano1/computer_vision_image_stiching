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

def main(img_view1,img_view2,correspond_points_in_A,correspond_points_in_B):
    """
    Assumes a relative path of two views of the same image (including the .jpg expension) and the path of a numpy array with coordinates of corresponding points (inludes .npy)
    Returns a matplotlib figure that displays the image view B with the provided corresponding points and the same image with the corresponding points estimated
    using homography matrix 

    Note: This prototype main function is meant to address the test required in question 2. Thereby, values are hardwired. 
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
    plt.title("Original Image")

    plt.scatter(cc1[:,0],cc1[:,1])

    #plt.show() # Keep this line commented out to show both images at once

    # Showing original image view B

    plt.figure()

    plt.imshow(crop2)
    plt.title("Original Image")

    plt.scatter(cc2[:,0],cc2[:,1])

    plt.show()

    print('Test start...')

    # Test begins by transposing points

    cc1_t = np.transpose(cc1)
    cc2_t = np.transpose(cc2)

    # Next step is scaling down the points
    cc1_t_scaled, cc1_min, cc1_max = scale_down(cc1_t)
    cc2_t_scaled, cc2_min, cc2_max = scale_down(cc2_t)

    # Next step is computing matrix h
    h_test1 = computeH(cc1_t_scaled,cc2_t_scaled)
    print(h_test1.shape)
    H_test1 = h_test1.reshape((3,3))

    print('\nThis is the homography matrix:\n')
    print(H_test1)

    # Next step is to transform scaled points 1 into homogeneous coordinates
    cc1_t_scaled_homog = to_homogeneous(cc1_t_scaled)

    # Then computing the estimated coordinates in view B from coordinates in biew A using the homography matrix
    estimated_cc2 = homogeneous_transform_coord(cc1_t_scaled_homog,H_test1)

    # Next transform the estimated homogenouscoordinates back to 2D coordinates
    estimated_cc2_2D_scaled = from_homogeneous(estimated_cc2)

    # Next step is scaling up the estimated coordinates
    estimated_cc2_2D = scale_back(estimated_cc2_2D_scaled, cc2_min, cc2_max)

    # the final step is displaying the image view B with both the original correspondence poins as well as the estimated points using homography matrix 
    
    fig = plt.figure()

    rows = 1
    columns = 2

    # Plot 1
    fig.add_subplot(rows, columns, 1) 
    
    # showing image 
    plt.imshow(crop2) 
    # plt.axis('off') 
    plt.title("Original Points") 

    plt.scatter(cc2_t[0,:],cc2_t[1,:])
    
    # Plot 2
    fig.add_subplot(rows, columns, 2) 
    
    # showing image 
    plt.imshow(crop2) 
    # plt.axis('off') 
    plt.title("Estimated Points") 

    plt.scatter(estimated_cc2_2D[0,:],estimated_cc2_2D[1,:],c='red')
    
    plt.show()
    
    return 0

if __name__ == "__main__":
    main(0,0,0,0)