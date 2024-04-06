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

# Loading images
PATH_NAUSICAA = '..\\Zambrano_Ricardo_ASN4_py\\input_nausicaa.jpg'
PATH_MONONOKE = '..\\Zambrano_Ricardo_ASN4_py\\input_mononoke.jpg'

img_raw_1 = io.imread(PATH_NAUSICAA)
img_raw_2 = io.imread(PATH_MONONOKE)

nausicaa = img_raw_1.copy()
mononoke = img_raw_2.copy()

views = [nausicaa,mononoke]

def main():

    print("-------------------------")
    print("Two windows will pop up. Each window displays a view of the same image. Using your mouse click on pair of corresponding points in each image.")
    print("-------------------------")
    print("Corrsponding points are pixels that below to the same part of the same object in each different view.  The object may have change position or being viewed from a different angle. ")
    print("-------------------------")
    print("-------------------------")
    print("Windows might pop up on top of each other, move them appart by using your mouse - Select the top of the window only")
    print("Once all the similar points are clicked, right-click on each image to close them.")
    print("-------------------------")

    pixel_coord_A, pixel_coord_B, data_coord_A, data_coord_B = getting_correspondences(nausicaa,mononoke)

    print("-------------------------\nCorrdinates in Image View A:\n")
    print(pixel_coord_A)
    print("-------------------------\nCorrdinates in Image View B:\n")
    print(pixel_coord_B)

if __name__ == "__main__":
    main()