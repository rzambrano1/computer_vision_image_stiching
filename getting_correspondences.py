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
from matplotlib.widgets import Cursor

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



def getting_correspondences(img1: npt.NDArray[np.uint8], img2: npt.NDArray[np.uint8]) -> Tuple[npt.NDArray,npt.NDArray,npt.NDArray,npt.NDArray]:
    """
    Collects corrdinates manually identified corresponding points from two views. Corresponding points are pixels that correspond to the same 
    part of a given object that is present in two different views.
    
    Uses mouse clicks to collect the coordinates dirctly from the matplotlib images.
    
    The number of correspondent points (clicks) on each image must match. Additionally, the client must provide at least four corresponding points.

    Given this is an client-driven function there are no guarantees the selected correspondent points actually belongs to the same object. The client's judgement
    will be used in desiding which point are correspondent.
    
    Input:
        img1, a matrix with dimmensions MxNx3 of data type uint8 that represents a view of an image
        img2, a matrix with dimmensions MxNx3 of data type uint8 that represents a view of an image
    
    Output:
        pixel_coordinates_img_A, corresponding point pixel coordinates in image view A 
        pixel_coordinates_img_B, corresponding point pixel coordinates in image view B
        data_coordinates_img_A, corresponding point data coordinates in image view A 
        data_coordinates_img_B, corresponding point data coordinates in image view B
    
    Parameters
    ----------
    img1 : np.ndarray [shape=(M,N,3)]
    img2 : np.ndarray [shape=(M,N,3)]
    
    Returns
    -------
    pixel_coordinates_img_A: np.ndarray [shape=(2,P)]
    pixel_coordinates_img_B: np.ndarray [shape=(2,P)]
    data_coordinates_img_A: np.ndarray [shape=(2,P)]
    data_coordinates_img_B: np.ndarray [shape=(2,P)]
    
    P: Number of selected correspondent pixel cordinates

    Throws
    ------
    Raises:AssertionError, if the number of clicked corresponding points in image view A is not equal to number of clicked corresponding points in image view B. Compare length of coordinates.
    Raises:AssertionError, if the number of clicked corresponding points is less than four

    Examples
    --------
    >>>
    >>>
    """

    # Initializing lists to store coordinates of corresponding pixels

    pixel_coordinates_img_A = []
    pixel_coordinates_img_B = []

    data_coordinates_img_A = []
    data_coordinates_img_B = []

    colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k']
    printColor = None

    # Showing Image View A
    
    plt.figure()
    plt.imshow(img1)
    plt.title("Image View A")
    
    def on_left_click(event):

        if event.button is MouseButton.LEFT:
            pixel_coordinates_img_A.append((event.x,event.y)) # Swap x <-> y in all
            data_coordinates_img_A.append((int(event.xdata),int(event.ydata)))

            print('Image View A ->',f'data coords {int(event.xdata)} {int(event.ydata)},',
                f'pixel coords {event.x} {event.y}')


    def on_right_click(event):
        if event.button is MouseButton.RIGHT:
            print('disconnecting callback - image view A')
            plt.disconnect(binding_id1)
            plt.close()

    binding_id1 = plt.connect('button_press_event', on_left_click)
    plt.connect('button_press_event', on_right_click)

    # Showing Image View B

    plt.figure()
    plt.imshow(img2)
    plt.title("Image View B")

    def on_left_click(event):

        if event.button is MouseButton.LEFT:
            pixel_coordinates_img_B.append((event.x,event.y))
            data_coordinates_img_B.append((int(event.xdata),int(event.ydata)))

            print('Image View B ->',f'data coords {int(event.xdata)} {int(event.ydata)},',
                f'pixel coords {event.x} {event.y}')


    def on_right_click(event):
        if event.button is MouseButton.RIGHT:
            print('disconnecting callback -  image view B')
            plt.disconnect(binding_id2)
            plt.close()

    binding_id2 = plt.connect('button_press_event', on_left_click)
    plt.connect('button_press_event', on_right_click)

    plt.show()

    assert len(pixel_coordinates_img_A) == len(pixel_coordinates_img_B), 'You must identify and click correspondent points on each image. Clicks on view A are different from clicks in view B'
    assert len(pixel_coordinates_img_A) > 3, " Provide at least four pairs of corresponfing points"

    return np.transpose(np.array(pixel_coordinates_img_A)), np.transpose(np.array(pixel_coordinates_img_B)), np.transpose(np.array(data_coordinates_img_A)), np.transpose(np.array(data_coordinates_img_B))

