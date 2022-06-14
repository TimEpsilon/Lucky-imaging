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
stsdas()
stsdas.analysis()
stsdas.analysis.restore()

R  = '\033[31m' # red
G  = '\033[32m' # green
P  = '\033[35m' # purple
W  = '\033[0m'  # white

plt.close('all')

def gauss_map(dx,dy,s):
    X,Y = np.indices((dx,dy))
    return np.exp(-((X-dx/2)**2 + (Y-dy/2)**2)/(2*s**2))


def save_normalized_copy(psf):
    '''
    Saves a copy of the given PSF, after apodizing and normalizing it

    Parameters
    ----------
    psf : str
        Path to the psf.

    '''
    # Copy of psf
    with fits.open(psf) as hdul:
        hdul.writeto(psf.replace(".fits", "_norm.fits")
                               ,output_verify='silentfix',overwrite=True)
        print(G + "Copy of " + psf + W)
        
    # Edit copy
    with fits.open(psf.replace(".fits", "_norm.fits"),mode="update") as hdul:
        x,y = np.shape(hdul[0].data)
        gauss = gauss_map(x,y,20)
        # hdul[0].data *= gauss
        # hdul[0].data /= hdul[0].data.sum()
        
    return

def deconvolution(img,psf,path):
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
    print(path)
    print(path.replace("mean","deconvolution"))
    
    if os.path.exists(path.replace("mean","deconvolution")):
        os.remove(path.replace("mean","deconvolution"))
    stsdas.analysis.restore.lucy(img, psf, path.replace("mean","deconvolution"), 1, 0, niter=10, limchisq=1e-10)
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
    # Getting img
    with fits.open(path) as img:
        d_img = img[0].data
        
    # Getting PSF
    psf_path = get_matching_psf(path)
    if psf_path is None:
        print(R + "No Match Found" + W)
        return
    
    # Normalize psf copy
    save_normalized_copy(psf_path)
    psf_path.replace(".fits","_norm.fits")
    
    with fits.open(psf_path) as psf:
        d_psf = psf[0].data
        
    # Saving deconvolution
    deconvolution(path,psf_path,path)
    
    # Getting deconv
    with fits.open(path.replace("mean", "deconvolution")) as hdul:
        deconv = hdul[0].data
    
    
    # reconv = convolve2d(deconv, psf,mode='same')
    
    fig,ax = plt.subplots(2,2)
    
    ax[0,0].imshow(img,norm=PowerNorm(0.5),cmap="afmhot")
    ax[0,0].set_title("Betelgeuse")
    ax[1,1].imshow(psf,norm=PowerNorm(0.5),cmap="afmhot")
    ax[1,1].set_title("Aldebaran")
    ax[1,0].imshow(deconv,norm=PowerNorm(0.5),cmap="afmhot")
    ax[1,0].set_title("Deconvolution")
    # ax[0,1].imshow(reconv,norm=PowerNorm(0.5),cmap="afmhot")
    ax[0,1].set_title("Reconvolution")
    
    
apply_deconvolution("/home/tdewacher/Documents/Stage/P94-2014-2015/Betelgeuse/0.008717/NB_4.05,/export/P94-2014-2015NACO.2015-02-07T01:07:04.840/P94-2014-2015NACO.2015-02-07T01:07:04.840_mean.fits")
    