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

pixel_coordinates_img_A = []
pixel_coordinates_img_B = []

data_coordinates_img_A = []
data_coordinates_img_B = []

def getting_correspondences(img1,img2):
    """
    
    """
    # Showing Image View A
    
    plt.figure()
    plt.imshow(img1)
    plt.title("Image View A")
    
    def on_move(event):
        if event.inaxes:
            print('Image View A ->',f'data coords {event.xdata} {event.ydata},',
                f'pixel coords {event.x} {event.y}')


    def on_click(event):
        if event.button is MouseButton.LEFT:
            print('disconnecting callback')
            plt.disconnect(binding_id1)
            plt.close()


    binding_id1 = plt.connect('motion_notify_event', on_move)
    plt.connect('button_press_event', on_click)

    # Showing Image View B

    plt.figure()
    plt.imshow(img2)
    plt.title("Image View B")

    def on_move(event):
        if event.inaxes:
            print('Image View B ->',f'data coords {event.xdata} {event.ydata},',
                f'pixel coords {event.x} {event.y}')


    def on_click(event):
        if event.button is MouseButton.LEFT:
            print('disconnecting callback')
            plt.disconnect(binding_id2)
            plt.close()


    binding_id2 = plt.connect('motion_notify_event', on_move)
    plt.connect('button_press_event', on_click)
    # pixel_coordinates = []

    # def on_left_click(event):
    #     if event.button is MouseButton.LEFT:
    #         pixel_coordinates.append((event.x,event.y))
    #         print(f'data coords {event.xdata} {event.ydata},',
    #             f'pixel coords {event.x} {event.y}')


    # def on_right_click(event):
    #     if event.button is MouseButton.RIGHT:
    #         print('disconnecting callback')
    #         plt.disconnect(binding_id)
    #         plt.close()
        
    #     return print(np.array(pixel_coordinates))

    # binding_id = plt.connect('button_press_event', on_left_click)
    # plt.connect('button_press_event', on_right_click)

    plt.show()

    return 1

