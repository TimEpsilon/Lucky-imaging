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
import pandas as pd
from scipy.optimize import curve_fit
from SynthPSF import getBand,createSynthPSF
from test_fit_UD import LDD_1D
from RotateImage import align_rotation
from scipy.ndimage import interpolation
stsdas()
stsdas.analysis()
stsdas.analysis.restore()

R  = '\033[31m' # red
G  = '\033[32m' # green
P  = '\033[35m' # purple
W  = '\033[0m'  # white
B =  '\033[34m' # blue

FWHM_TO_SIGMA = 2*np.sqrt(2*np.log(2))

plt.close('all')

def gauss_map(dx,dy,s):
    X,Y = np.indices((dx,dy),dtype=float)
    return np.exp(-((X-dx/2+0.5)**2 + (Y-dy/2+0.5)**2)/(2*s**2))

def crop_image(img,dx,dy):
    '''
    Crops the given image, centered on the middle, to the given dimension.
    If the image is a datacube (supposed of shape (T,Y,X)), the datacube will be cropped

    Parameters
    ----------
    img : 2D/3D array
        image to crop.
    dx : int
        width.
    dy : int
        height.

    Returns
    -------
    c_img : 2D/3D array
        The cropped image.

    '''
    if len(np.shape(img)) == 3:
        for i in range(np.shape(img)[0]):
            img[i,:,:] = crop_image(img[i,:,:], dx, dy)
        return img
    
    else:
        Dy,Dx = np.shape(img)
    
        return img[int(Dy/2-dy/2):int(Dy/2+dy/2) , int(Dx/2-dx/2):int(Dx/2+dx/2)]


def save_normalized_copy(psf,n_img,bet_path):
    '''
    Saves a copy of the given PSF, after apodizing and normalizing it

    Parameters
    ----------
    psf : str
        Path to the psf.
    n_img : 2D array
        The image we want to deconvolve
        
    Returns
    -------

    '''
    # True PSF
    createTruePSF(psf)
    psf = psf.replace("mean", "true")
    
    # Rotate
    align_rotation(psf, bet_path)
    psf = psf.replace("true","rot")
    
    
    # Copy of psf
    with fits.open(psf) as hdul:
        img = hdul[0].data
        hdr = hdul[0].header
        
        
        dy,dx = np.shape(img)
        dx = min(dx,np.shape(n_img)[0])
        dy = min(dy,np.shape(n_img)[1])
        
        img = crop_image(img, dx, dy)
        n_img = crop_image(n_img, dx, dy)
        
        img[img<0]=np.min(img[img>0])
        img[n_img<=0]=np.min(img[img>0])
        
        # Apodizing
        gauss = gauss_map(dy,dx,60)
        img *= gauss
        img /= hdul[0].data.max()
        
        img = np.nan_to_num(img)
        
        n_hdul = fits.PrimaryHDU(img,header=hdr)
        n_hdul.writeto(psf.replace("rot","norm")
                               ,output_verify='silentfix',overwrite=True)
        
    print(G + "Copy of " + psf + W)      
    return

def save_pos_copy(path,psf):
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
    
    with fits.open(psf) as hdul:
        n_img = hdul[0].data
        
        
    dx,dy = np.shape(n_img)
    gauss = gauss_map(dx,dy,60)
    img *= gauss
    img[img<0] = np.min(img[img>0])
    
    n_hdul = fits.PrimaryHDU(img,header=hdr)
    n_hdul.writeto(path.replace("mean","norm")
                           ,output_verify='silentfix',overwrite=True)
    print(G + "Copy of " + path + W)


def deconvolution(img,psf,n=50):
    '''
    Attempts a deconvolution on img using a given psf.

    Parameters
    ----------
    img : str
        Image to deconvol.
    psf : str
        point spread function.
    n : int
        Number of iterations
    '''

    if os.path.exists(img.replace("norm","deconvolution-temp")):
        os.remove(img.replace("norm","deconvolution-temp"))
    stsdas.analysis.restore.lucy(img, psf, img.replace("norm","deconvolution-temp"), 40, 0,
                                 niter=n, limchisq=1e-10,verbose=False,nsave=2)
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
    
    deconv_match = path[:-len(path.split('/')[-1])] + "deconvolution_match.txt"
    if os.path.exists(deconv_match) :
        with open(deconv_match,"r") as read_file:
            corr_path = read_file.readlines()[0]
            return corr_path.replace("\n","")
    
    ald_path = path[:-len(path.split('/')[-6] + "/" +
                                 path.split('/')[-5] + "/" +
                                 path.split('/')[-4] + "/" +
                                 path.split('/')[-3] + "/" +
                                 path.split('/')[-2] + "/" + 
                                 path.split('/')[-1])] + "Aldebaran/"
    
    good_list = []
    for t in os.listdir(ald_path):
        filt_path = ald_path + t + "/" + filt
        if os.path.exists(filt_path + "/export"):
            name_list = os.listdir(filt_path + "/export/")
            for name in name_list:
                if "bad" not in name and os.path.exists(filt_path + "/export/" + name + "/" + name + "_mean.fits"):
                    good_list.append(filt_path + "/export/" + name + "/" + name + "_mean.fits")
           
    if len(good_list) == 0:
        return
    
    if len(good_list) == 1:
        corr_path = good_list[0]
        print(G + "Path found ! " + corr_path + W)
        return corr_path
    
    fig,ax = plt.subplots(len(good_list))
    for i in range(len(good_list)):
        f = good_list[i]
        
        # plot image
        with fits.open(f) as hdul:
            if len(np.shape(hdul[0].data)) == 3:
                img = hdul[0].data[0,:,:]
            else:
                img = hdul[0].data
            ax[i].imshow(img,norm=PowerNorm(0.3),cmap="afmhot")
            ax[i].set_title(str(i) + " : " + f.split('/')[-1])
    
    plt.show(block=False)
    
    # Choose image to use
    _id = len(good_list)+1
    while _id > len(good_list):
        _id = input("Please input Aldebaran id to use : ")
    
    corr_path = good_list[_id]
    print(G + "Path found ! " + corr_path + W)
    
    with open(deconv_match,"w") as write_file:
        write_file.write(corr_path)
            
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
    save_pos_copy(path,psf_path)
    
    # Getting img
    with fits.open(path.replace("mean","norm")) as img:
        d_img = img[0].data
        hdr_img = img[0].header
        
    
    # Normalize psf copy
    save_normalized_copy(psf_path,d_img,path)
    psf_path.replace("mean","norm")
    
    with fits.open(psf_path.replace("mean","norm")) as psf:
        d_psf = psf[0].data
        hdr_psf = psf[0].header
        
    # Dimension
    dx,dy = np.shape(d_img)
    dx1,dy1 = np.shape(d_psf)
    dl = min(dx,dx1,dy,dy1)
    
    # Cropping
    d_img = crop_image(d_img, dl, dl)
    d_psf = crop_image(d_psf, dl, dl)
    
    # Rescaling
    d_img = interpolation.zoom(d_img,4)
    d_psf = interpolation.zoom(d_psf,4)
    
    # Change pixel size
    hdr_img["CD2_2"] /= 4
    hdr_img["CD1_1"] /= 4
    hdr_psf["CD2_2"] /= 4
    hdr_psf["CD1_1"] /= 4
    
    
    hdul_psf = fits.PrimaryHDU(d_psf,header=hdr_psf)
    hdul_psf.writeto(psf_path.replace("mean","norm"),overwrite=True,output_verify="silentfix")
    
    hdul_bet = fits.PrimaryHDU(d_img,header=hdr_img)
    hdul_bet.writeto(path.replace("mean","norm"),overwrite=True,output_verify="silentfix")
    
        
    # Saving deconvolution
    deconvolution(path.replace("mean","norm"),psf_path.replace("mean","norm"))
    
    # Getting deconv
    with fits.open(path.replace("mean", "deconvolution-temp"),ignore_missing_end=True) as hdul:
        deconv = hdul[0].data
        hdr = hdul[0].header
    
    hdul = fits.PrimaryHDU(deconv,header=hdr)
    hdul.writeto(path.replace("mean", "deconvolution-temp"),overwrite=True,output_verify="silentfix")
    
    # Plot iterations
    directory = path[:-len(path.split("/")[-1])]
    i,d,err = getDiameterAndIteration(directory)
    pxToMas = hdr["CD2_2"]*3600*1000
    plt.figure()
    # plt.scatter(i,0.79*(np.array(d)*pxToMas)+21,marker='+',color='red')
    plt.errorbar(i,0.79*(np.array(d)*pxToMas)+21,yerr=0.79*(np.array(err)*pxToMas),fmt=".",color="gray")
    plt.hlines(43.7,0,50)
    plt.xlabel("Iterations")
    plt.ylabel("Angular Diameter (mas)")
    plt.title(path.split("/")[-1])
    plt.show(block=False)
    
    
    # Datacube of every iteration
    
    it, imgs = [],[]
    with fits.open(path.replace("mean","deconvolution-temp")) as hdul:
        final = hdul[0].data
        imgs.append(final)
        it.append(50)
    
    os.remove(path.replace("mean","deconvolution-temp"))
        
    for f in os.listdir(directory):
        if "temp" in f:
            with fits.open(directory+f) as hdul:
                img = hdul[0].data
                hdr1 = hdul[0].header
                
                imgs.append(img)
                it.append(hdr1["NO_ITER"])
            os.remove(directory+f)
               
    index = range(len(it))
    index = [x for _,x in sorted(zip(it,index))]
    imgs = [imgs[i] for i in index]
    
    with fits.open(path.replace("mean","norm")) as hdul:
        imgs.insert(0,hdul[0].data)
    
    data = np.stack((imgs))
    
    
    hdul = fits.PrimaryHDU(data,header=hdr)
    hdul.writeto(path.replace("mean", "deconvolution"),overwrite=True,output_verify="silentfix")
    
    return

def Gauss2D(XY,A,fwhm):
    '''
    2D gauss function

    Parameters
    ----------
    XY : [X,Y]
        Array containing the values (array or single) of x and y.
    A : nbr
        Max value.
    fwhm : nbr
        Full Width at Half max.
    x0 : nbr
        x component of the center.
    y0 : nbr
        y component of the center.
    c : nbr
        offset on the z axis.

    Returns
    -------
    Z: array/nbr
        The value at (X,Y)

    '''
    x,y = XY
    s = fwhm/FWHM_TO_SIGMA
    
    return A * np.exp(-((x)**2 + (y)**2)/(2*s**2))
    
def Gauss2D_ravel(XY,A,fwhm):
    return np.ravel(Gauss2D(XY,A,fwhm))

def getDiameterAndIteration(directory):
    '''
    Returns the diameter and the corresponding iteration

    Parameters
    ----------
    directory : path to the directory

    Returns
    -------
    i : iteration
    d : diameter

    '''
    i = []
    d = []
    err = []
    for f in os.listdir(directory):
        if "temp" in f and ".fits" in f:
            with fits.open(directory+f) as hdul:                
                img = hdul[0].data
                hdr = hdul[0].header
                
                # Getting Diameter;
                X,Y = np.indices(np.shape(img),dtype=float)
                X -= np.shape(X)[0]/2 - 0.5
                Y -= np.shape(Y)[1]/2 - 0.5
                opt, cov = curve_fit(Gauss2D_ravel, (X, Y), np.ravel(img), p0=[np.max(img), 5])
                
                # append
                i.append(int(hdr["NO_ITER"]))
                d.append(opt[1])
                err.append(np.sqrt(cov[1,1]))           
    return i,d,err

def createTruePSF(path):
    with fits.open(path) as hdul:
        hdr = hdul[0].header
        
    pxToDeg = hdr["CD2_2"]
    filt = path.split('/')[-4]
    i = getBand(filt)
    true_psf = main+"synthetic_psf/"+i+"-"+str(pxToDeg)+".fits"
    
    if not os.path.exists(true_psf):
        createSynthPSF((10,10),pxToDeg*1000*3600,i+"-"+str(pxToDeg))
        
    if os.path.exists(path.replace("mean","true")):
        os.remove(path.replace("mean", "true"))
    stsdas.analysis.restore.lucy(path, true_psf, path.replace("mean","true"), 40, 0,
                                 niter=1, limchisq=1e-10,verbose=False,nsave=5)
        


# =============================================================================
#                                    MAIN
# =============================================================================


main = "/home/tdewacher/Documents/Stage/" 
folders = ["P82-2008-2009"]

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
    print(G + "Working on " + f + W)
    
    # Deconvolution
    apply_deconvolution(f+"_mean.fits")
    
    print(B + str(i+1) + "/" + str(len(clean_list)) + " \n" + W)
    
    plt.show(block=False)