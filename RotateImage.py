#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul  8 15:57:18 2022

@author: tdewacher
"""

import numpy as np
from scipy.ndimage import rotate
import os
from astropy.io import fits
from matplotlib.colors import PowerNorm
import matplotlib.pyplot as plt

def crop(img,amount):
    '''
    Crops the given image by a given amount of pixels in every direction

    Parameters
    ----------
    img : 2D array
    amount : int

    Returns
    -------
    cropped : 2D array

    '''
    dx,dy = np.shape(img)
    cropped = img[amount:dx-amount,amount:dy-amount]
    return cropped

def align_rotation(ald,bet):
    '''
    Rotates the picture of Aldebaran in order to align it with Betelgeuse and saves it to a fits

    Parameters
    ----------
    ald : str
        Path to the PSF of Aldebaran.
    bet : str
        Path to the corresponding Betelgeuse image.

    '''
    
    # Getting images and hdrs
    with fits.open(ald) as hdul:
        img_ald = hdul[0].data
        hdr_ald = hdul[0].header
    with fits.open(bet) as hdul:
        img_bet = hdul[0].data
        hdr_bet = hdul[0].header
        
    # Getting the angle
    alpha = -(hdr_bet["HIERARCH ESO TEL PARANG START"] - hdr_ald["HIERARCH ESO TEL PARANG START"])
    print(alpha)
    
    rotated = rotate(img_ald,alpha,reshape=False)
    
    # Crop image
    i = 1
    while len(rotated[rotated == 0]) != 0:
        rotated = crop(rotated,i)
        
    # Saving the rotated image
    hdul = fits.PrimaryHDU(rotated,hdr_ald)
    hdul.writeto(ald.replace("true","rot"),overwrite=True,output_verify="silentfix")
    return
