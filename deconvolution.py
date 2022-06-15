#!/usr/bin/env python2.7
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 13 15:46:03 2022

@author: tdewacher
"""

import os
from glob import glob
import datetime
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm, PowerNorm
from astropy.io import fits
from pyraf.iraf import stsdas
from scipy.signal import convolve2d
import time
stsdas()
stsdas.analysis()
stsdas.analysis.restore()

R  = '\033[31m' # red
G  = '\033[32m' # green
P  = '\033[35m' # purple
W  = '\033[0m'  # white
B =  '\033[34m' # blue

# plt.close('all')

def gauss_map(dx,dy,s):
    X,Y = np.indices((dx,dy))
    return np.exp(-((X-dx/2)**2 + (Y-dy/2)**2)/(2*s**2))


def save_normalized_copy(psf,n_img):
    '''
    Saves a copy of the given PSF, after apodizing and normalizing it

    Parameters
    ----------
    psf : str
        Path to the psf.
    n_img : 2D array
        The image we want to deconvolve

    '''
    # Copy of psf
    with fits.open(psf) as hdul:
        img = hdul[0].data
        hdr = hdul[0].header
        
        img[img<0]=0
        img[n_img<=0]=0
        
        # Apodizing
        x,y = np.shape(hdul[0].data)
        gauss = gauss_map(x,y,10)
        img *= gauss
        img /= hdul[0].data.sum()
        
        img = np.nan_to_num(img)
        
        n_hdul = fits.PrimaryHDU(img,header=hdr)
        n_hdul.writeto(psf.replace("mean","norm")
                               ,output_verify='silentfix',overwrite=True)
        print(G + "Copy of " + psf + W)      
    return

def save_pos_copy(path):
    '''
    Saves a copy of the given image, after setting any negative value to 0

    Parameters
    ----------
    path : str
        Path to the img.

    '''
    # Copy of img
    with fits.open(path) as hdul:
        img = hdul[0].data
        hdr = hdul[0].header
        
        # img[img<0]=0
        
        img = np.nan_to_num(img)
        
        n_hdul = fits.PrimaryHDU(img,header=hdr)
        n_hdul.writeto(path.replace("mean","norm")
                               ,output_verify='silentfix',overwrite=True)
        print(G + "Copy of " + path + W)      
    return


def deconvolution(img,psf,n=40):
    '''
    Attempts a deconvolution on img using a given psf.

    Parameters
    ----------
    img : str
        Image to deconvol.
    psf : str
        point spread function.
    path : str
        Location to save file

    '''

    if os.path.exists(img.replace("norm","deconvolution")):
        os.remove(img.replace("norm","deconvolution"))
    stsdas.analysis.restore.lucy(img, psf, img.replace("norm","deconvolution"), 1, 0, niter=n, limchisq=1e-10,verbose=False)
    return

def get_matching_psf(path):
    '''
    Returns the path to corresponding Aldebaran mean image, if it exists.

    Parameters
    ----------
    path : str
        path to the Betelgeuse mean image.

    Returns
    -------
    corr_path : str
        The corresponding (same epoch and filter) Aldebaran path.
        None if not exist.

    '''
    corr_path = None
    filt = path.split('/')[-4]
    ald_path = path[:-len(path.split('/')[-6] + "/" +
                                 path.split('/')[-5] + "/" +
                                 path.split('/')[-4] + "/" +
                                 path.split('/')[-3] + "/" +
                                 path.split('/')[-2] + "/" + 
                                 path.split('/')[-1])] + "Aldebaran/"
    
    for t in os.listdir(ald_path):
        filt_path = ald_path + t + "/" + filt
        if os.path.exists(filt_path + "/export"):
            name_list = os.listdir(filt_path + "/export/")
            for name in name_list:
                if "bad" not in name:
                    break
            corr_path = filt_path + "/export/" + name + "/" + name + "_mean.fits"
    print(G + "Path found ! " + corr_path + W)
    return corr_path

def apply_deconvolution(path):
    '''
    Tries to deconvolve the Betelgeuse image at the given path.

    Parameters
    ----------
    path : str
        Path to the Betelgeuse mean image.

    '''  
    # Getting PSF
    psf_path = get_matching_psf(path)
    if psf_path is None:
        print(R + "No Match Found" + W)
        return
    
    # Creating positive img
    save_pos_copy(path)
    
    # Getting img
    with fits.open(path.replace("mean","norm")) as img:
        d_img = img[0].data
        
    
    # Normalize psf copy
    save_normalized_copy(psf_path,d_img)
    psf_path.replace("mean","norm")
    
    with fits.open(psf_path.replace("mean","norm")) as psf:
        d_psf = psf[0].data
        
    # Saving deconvolution
    deconvolution(path.replace("mean","norm"),psf_path.replace("mean","norm"))
    
    # Getting deconv
    with fits.open(path.replace("mean", "deconvolution")) as hdul:
        deconv = hdul[0].data
    
    
    reconv = convolve2d(deconv, d_psf,mode='same')
    
    fig,ax = plt.subplots(2,2)
    
    ax[0,0].imshow(np.nan_to_num(d_img),norm=PowerNorm(0.3),cmap="afmhot",interpolation='bicubic')
    ax[0,0].set_title("Betelgeuse : " + path.split('/')[-2])
    ax[1,1].imshow(np.nan_to_num(d_psf),norm=PowerNorm(0.3),cmap="afmhot",interpolation='bicubic')
    ax[1,1].set_title("Aldebaran" + path.split('/')[-4])
    ax[1,0].imshow(deconv,norm=PowerNorm(0.3),cmap="afmhot",interpolation='bicubic')
    ax[1,0].set_title("Deconvolution")
    ax[0,1].imshow(reconv,norm=PowerNorm(0.3),cmap="afmhot",interpolation='bicubic')
    ax[0,1].set_title("Reconvolution")
    
    plt.draw()
    return 

main = "/home/tdewacher/Documents/Stage/" 
folders = ["P82-2008-2009","P88-2011-2012","P90-2012-2013","P94-2014-2015"]

start = time.time()
clean_list = []

for f in folders:
    for obj in os.listdir(main+f):
        if obj != "Betelgeuse": 
            continue
        for times in os.listdir(main+f+"/"+obj):
            for filt in os.listdir(main+f+"/"+obj+"/"+times):
                if os.path.exists(main+f+"/"+obj+"/"+times+"/" + filt + "/export"):
                    for name in os.listdir(main+f+"/"+obj+"/"+times+"/" + filt + "/export"):
                        if "_bad" in name:
                            continue
                        file = main+f+"/"+obj+"/"+times+"/" + filt + "/export/" + name +"/"+name
                        clean_list.append(file)

print(B + str(len(clean_list)) + " files to work on" + W)

for i in range(len(clean_list)):
    f = clean_list[i]
    apply_deconvolution(f+"_mean.fits")
    print(B + str(i+1) + "/" + str(len(clean_list)) + " \n" + W)
    

print(G + "Temps écoulé : " + str(int(time.time() - start)) + "s" + W)

plt.show()