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
# None

def computeH(t1: npt.NDArray, t2: npt.NDArray)-> npt.NDArray:
    """
    Takes a set of corresponding image points t1, t2 (both t1 and t2 should be 2xN matrices) and computes the associated 3 x 3 homography matrix H. The client should
    provide a list of P ≥ 4 pairs of corresponding points from the two views, where each point is specified with its own 2d image coordinates. 

    Input:
        t1, a numpy array with dimension 2xP with coordinates for corresponding points from image view A, P should be greater than 3. Top row are x coordinates, bottom row are y coordinates
        t2, a numpy array with dimension 2xP with coordinates for corresponding points from image view B, P should be greater than 3. Top row are x coordinates, bottom row are y coordinates
    
    Output:
        homography_matrix, the 3x3 homography matrix H associated with the two list of corresponding points
    
    Parameters
    ----------
    t1 : np.ndarray [shape=(2,P)]
    t2 : np.ndarray [shape=(2,P)]
    
    P > 3

    Returns
    -------
    homography_matrix: np.ndarray [shape=(3,3)]

    Throws
    ------
    Raises:AssertionError, if the dimensions of either t1 or t2 are not equal
    Raises:AssertionError, if the number of rows of either t1 or t2 are greater than 2
    Raises:AssertionError, if the number of columns of either t1 or t2 are not greater than 4 

    Examples
    --------
    >>>
    >>>
    """
    assert t1.shape == t2.shape, 'The nuber of corresponding points in t1 must be equal to the number of corresponding points in t2.'
    assert (t1.shape[0]==2) and (t2.shape[0]==2), 'The expected dimensions in both t1 and t2 are 2, x on the top row and y in the bottom row. The client provided more than 2D'
    assert (t1.shape[1]>3) and (t2.shape[1]>3), 'The number of corresponding pairs of coordinates must be equal or greater than 4.'

    return homography_matrix
    