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

    for j in range(w):
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

    for indx in range(coord.shape[1]):
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
              corr_points_input: npt.NDArray = None,
              corr_points_ref: npt.NDArray = None
              )-> Tuple[npt.NDArray[np.uint8],npt.NDArray[np.uint8]]:
    """
    Assumes as input an image inputIm, a reference image refIm, and a 3x3 homography matrix H. corr_points_input and corr_points_ref are corresponding points 
    used to calculate H
    Returns two images as outputs. The first image, warpIm, is the input image inputIm warped according to H to fit within the frame of the reference image refIm. 
    The second output image, mergeIm, is a single mosaic image with a larger field of view containing both input images

    Input:
        inputIm, a matrix with dimmensions MxNx3 of data type uint8 that represents a view of an image
        refIm, a matrix with dimmensions MxNx3 of data type uint8 that represents a view of an image
        H, homography_matrix, the 3x3 homography matrix H associated with the two images
        corr_points_input, a numpy array with dimension 2xP with coordinates for corresponding points from inputIm (image view A)
        corr_points_ref, a numpy array with dimension 2xP with coordinates for corresponding points from refIm (image view B)
    
    Output:
        warpIm, a matrix with dimmensions MxNx3 of data type uint8 that represents inputIm warped
        mergeIm, a matrix with dimmensions MxNx3 of data type uint8 that represents a mosaic composed of inputIm and refIm
    
    Parameters
    ----------
    inputIM : np.ndarray [shape=(M,N,3)]
    refIm : np.ndarray [shape=(M,N,3)]
    H : np.ndarray [shape=(3,3)]
    corr_points_input : np.ndarray [shape=(2,P)]
    corr_points_ref : np.ndarray [shape=(2,P)]
    
    P > 3

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

    # The min and max values for inputIm are required
    _, t1_test_min, t1_test_max, t2_test_min, t2_test_max = pre_process_correspondences(corr_points_input,corr_points_ref)

    # Since the HW requires inverse warp we need the inverse of the homography matrix
    H_inv = np.linalg.inv(H)

    # Recording dimensions of image views
    input_height, input_width, ref_channels = inputIm.shape
    ref_height, ref_width, ref_channels = refIm.shape

    # Then record the frame of each image unwarped.
    # The corners in order are: origin, opposite to origin, bottom_left, top_right

    frame_input =np.array([
        [0, input_width, 0,input_width],
        [0, input_height, input_height, 0]
    ])

    frame_ref = np.array([
        [0, ref_width, 0, ref_width],
        [0, ref_height, ref_height, 0]
    ])

    # Calculate corners for bounding box
    bounding_box = calculate_inverse_frame(frame_ref, H_inv, t1_test_min, t1_test_max)

    # Finding the dimensions of the bounding box
    bounding_box_width = np.round(np.max(bounding_box[0,:])).astype(int)
    bounding_box_height = np.round(np.max(bounding_box[1,:])).astype(int)

    # Creating an array to record the warped points
    empty_warpIm = np.zeros((bounding_box_height,bounding_box_width,3))

    warpIm = inverse_warping(refIm, H_inv, empty_warpIm, t1_test_min, t1_test_max)

    return warpIm, mergeIm