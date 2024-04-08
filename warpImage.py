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

#################################################
# Main function or Major function of the module #
#################################################

def warpImage(inputIm: npt.NDArray[np.uint8], refIm:npt.NDArray[np.uint8], H: npt.NDArray)-> Tuple[npt.NDArray[np.uint8],npt.NDArray[np.uint8]]:
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

    return waprIm, mergeIm