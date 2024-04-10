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
# Functions Import
from computeH import scale_down
from computeH import scale_back
from computeH import to_homogeneous
from computeH import from_homogeneous
from computeH import homogeneous_transform_coord
from computeH import computeH

####################
# Helper Functions #
####################

def pre_process_correspondences(t1: npt.NDArray, t2: npt.NDArray)->Tuple[npt.NDArray,np.double,np.double,np.double,np.double]:
    """
    Assumes two arrays of 2xP with correspondence points.
    Returns the homography matrix and the parameters to scale back the points once the estimated destinations points are 
    calculated using homography. 
    """
    # Scaling correspondence points
    t1_scaled, t1_min, t1_max = scale_down(t1)
    t2_scaled, t2_min, t2_max = scale_down(t2)
    
    # Computing H
    h = computeH(t1_scaled,t2_scaled)
    H = h.reshape((3,3))
    
    return H, t1_min, t1_max, t2_min, t2_max

def calculate_inverse_frame(frame_corners,H_inv,r_i_min,r_i_max): # r_ref_min and r_ref_max are not needed
    """
    Assumes a numpy array of shape (2,4) with four corners belonging to an image view B and the inverse of an homography matrix
    Returns coordinates of frame for warping and inmage.
    """
    
    # Step 1: scaling down
    scaled_frame_corners, rmin, rmax = scale_down(frame_corners)
    # Step 2: Transforming scaled coordinates into homogeneous coordinates 
    frame_scaled_homog = to_homogeneous(scaled_frame_corners)
    # Step 3: Estimating value with inverse homography matrix
    estimated_frame = homogeneous_transform_coord(frame_scaled_homog,H_inv)
    # Step 4: Converting estimated homogeneous coordinates into 2D coordinates
    estimated_frame_scaled =  from_homogeneous(estimated_frame)
    # Step 5: Scale back
    estimated_frame_corners = scale_back(estimated_frame_scaled,r_i_min,r_i_max)
    
    return estimated_frame_corners


def inverse_warping(im_view_b, H_inv, box, r_i_min, r_i_max):
    """
    Assumes an RGB image, inverse homography matrix, initialized array to record the warped image, and
    the min and max coordinates of the image view A.
    Returns a warped image
    """
    w,h,ch = im_view_b.shape
    warped = np.zeros_like(box)

    # First generate all coordinates in image view B

    # Initialize coord
    coord = np.array([
                [0],
                [0]
            ])

    print('Initializing coordinates array...\nMay take a few minutes...')
    for j in tqdm(range(w)):
        for i in range(h):
            #Create a coordinate
            new_coord = np.array([
                [i],
                [j]
            ])
            coord = np.hstack((coord,new_coord))

    coord = coord[:,1:]
    
    # Calculate position coordinate using homography
    
    warped_coord = calculate_inverse_frame(coord, H_inv, r_i_min, r_i_max)
    
    # Populating RGB intensities in warped

    print('Populating RGB intensities in warped images...')
    for indx in tqdm(range(coord.shape[1])):
        j,i = tuple(coord[:,indx])
        j_prime,i_prime, = tuple(warped_coord[:,indx].astype(int))
        try:
            warped[int(i_prime), int(j_prime), :] = im_view_b[i,j,:]
        except:
            if (int(i_prime) > warped.shape[0]-1) and (int(j_prime) < warped.shape[1]-1):
                i_prime = warped.shape[0]-1
            elif (int(j_prime) > warped.shape[1]-1) and (int(i_prime) < warped.shape[0]-1):
                j_prime = warped.shape[1]-1
            else:
                i_prime = warped.shape[0]-1
                j_prime = warped.shape[1]-1
            warped[int(i_prime), int(j_prime), :] = im_view_b[i,j,:]

    return warped.astype(np.uint8)

#################################################
# Main function or Major function of the module #
#################################################

def warpImage(inputIm: npt.NDArray[np.uint8], 
              refIm:npt.NDArray[np.uint8], 
              H: npt.NDArray,
              )-> Tuple[npt.NDArray[np.uint8],npt.NDArray[np.uint8]]:
    """
    Assumes as input an image inputIm, a reference image refIm, and a 3x3 homography matrix H.
    Returns two images as outputs. The first image, warpIm, is the input image inputIm warped according to H to fit within the frame of the reference image refIm. 
    The second output image, mergeIm, is a single mosaic image with a larger field of view containing both input images

    Input:
        inputIm, a matrix with dimmensions MxNx3 of data type uint8 that represents a view of an image
        refIm, a matrix with dimmensions MxNx3 of data type uint8 that represents a view of an image
        H, homography_matrix, the 3x3 homography matrix H associated with the two images
    
    Output:
        warpIm, a matrix with dimmensions MxNx3 of data type uint8 that represents inputIm warped
        mergeIm, a matrix with dimmensions MxNx3 of data type uint8 that represents a mosaic composed of inputIm and refIm
    
    Parameters
    ----------
    inputIM : np.ndarray [shape=(M,N,3)]
    refIm : np.ndarray [shape=(M,N,3)]
    H : np.ndarray [shape=(3,3)]

    Returns
    -------
    warpIm : np.ndarray [shape=(M,N,3)]
    mergeIm : np.ndarray [shape=(M,N,3)]
    
    Throws
    ------
    Raises:AssertionError, if the dimensions of either inpudIm or refIm are not 3-channel arrays
    Raises:AssertionError, if the homography matrix has not 3x3 dimensions

    Examples
    --------
    >>>
    >>>
    """
    assert inputIm.shape[2] == 3, 'Expected a 3-channel array.'
    assert refIm.shape[2] == 3, 'Expected a 3-channel array.'
    assert H.shape == (3,3), 'The homography matrix parameter (H) must be a (3,3) array.'
    
    # This code follows the same approach as warpImage_old.py (The one I developed from the scratch with the book)
    # This approach was calculating a frame, creating an empty image, and then filling the image. In the case of the warped image
    # using the homography matrix to translate points and in the case of the merged image taking points from both images. 
    # However, I had to incorporate adaptations from explainers and examples to make it work. I put the steps to showcase the approach was similar to 
    # warpImage_old.py

    # Since the HW requires inverse warp we need the inverse of the homography matrix
    print('Calculating inverse of homography matrix...')
    H_inv = np.linalg.inv(H)

    print('Creating mosaic...')
    # Recording dimensions of image views
    input_height, input_width, c = inputIm.shape
    output_height, output_width, c = refIm.shape

    # Initializing corners
    min_x = float("inf")
    min_y = float("inf")
    max_x = float("-inf")
    max_y = float("-inf")

    # Calculate corners for bounding box
    cornersi =  [(0,0), (input_height, input_width), (0, input_width), (input_height, 0)]
    cornerso =  [(0,0), (input_height, input_width), (0, input_width), (input_height, 0)]

    for i,j in cornersi:
            x, y, w = np.matmul(H, [j , i, 1])
            x = x/w
            y = y/w
            if x > max_x:
                max_x = int(x)
            if x < min_x:
                min_x = int(x)
            if y > max_y:
                max_y = int(y)
            if y < min_y:
                min_y = int(y)
    
    # Creating an empty array to record the warped points
    warpIm = np.zeros((max_y - min_y,max_x - min_x, 3))

    # Filling the array with the warped image
    for i in range(0, max_x - min_x):
        for j in range (0, max_y - min_y):
            x, y, w = np.matmul(H_inv, [i + min_x, j + min_y, 1])
            x = int(x/w)
            y = int(y/w)
            a = 0
            b = 0
            c = 0
            if not (y < 0 or y >= input_height or x < 0 or x >= input_width):
                a, b, c = inputIm[y, x, :]
            warpIm[j, i, :] = [a/255, b/255, c/255]
    oldx = min_x
    oldy = min_y
    oldmx = max_x
    oldmy = max_y

    # Calculating size of merged image
    for i,j in cornerso:
        if j > max_x:
            max_x = int(j)
        if j < min_x:
            min_x = int(j)
        if i > max_y:
            max_y = int(i)
        if i < min_y:
            min_y = int(i)
    
    # Creating an empty array to record the merged image
    mergeIm = np.zeros(((max_y - min_y),(max_x - min_x), 3))

    # Final step, filling the merged image frame
    for i in range(min_x, max_x):
        for j in range (min_y, max_y):
            a = 0
            b = 0
            c = 0
            if not (j < oldy or j >= oldmy or i < oldx or i >= oldmx):
                a, b, c = warpIm[j - oldy, i - oldx, :]
                if a == 0.0 or b == 0.0 or c == 0.0:
                    if not (j < 0 or j >= output_height or i < 0 or i >= output_width):
                        a, b, c = refIm[j, i, :]/255
            else:
                if not (j < 0 or j >= output_height or i < 0 or i >= output_width):
                    a, b, c = refIm[j, i, :]/255
            mergeIm[j - min_y, i- min_x, :] = [a, b, c]

    return warpIm, mergeIm