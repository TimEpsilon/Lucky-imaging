#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun  9 09:58:47 2022

@author: tdewacher
"""

import numpy as np 
from image_registration import chi2_shift
from image_registration.fft_tools import shift
import os
from astropy.io import fits
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import pandas as pd
from scipy.signal import convolve2d
import time
import datetime
from matplotlib.colors import PowerNorm

FWHM_TO_SIGMA = 2*np.sqrt(2*np.log(2))
R  = '\033[31m' # red
G  = '\033[32m' # green
P  = '\033[35m' # purple
W  = '\033[0m'  # white
B =  '\033[34m' # blue

plt.close('all')

def align(offset,image):
    """
    Aligns offset to image.

    Parameters
    ----------
    offset : 2D array
        The image to align.
    offset : 2D array
        The reference image to align to.

    Returns
    -------
    corrected : 2D array
        The offset image after alignement with image
    """
    xoff, yoff, exoff, eyoff = chi2_shift(image, offset,upsample_factor=3)
    
    return shift.shiftnd(offset,[-yoff,-xoff])

def align_datacube(datacube):
    '''
    Aligns every image to a centered gaussian.

    Parameters
    ----------
    datacube : 3D array [time,x,y]
        The set of images to align.

    Returns
    -------
    aligned: 3D array [time,x,y]
        The aligned set of images

    '''
    ti = time.time()
    
    if len(np.shape(datacube)) == 3:
        n = np.shape(datacube)[0]
        
        ref = gauss_map(np.shape(datacube)[-2], np.shape(datacube)[-1], np.max(datacube))
        # For every frame
        for i in range(n):
            datacube[i,:,:] = align(datacube[i,:,:],ref)
            if i % 500 == 0 and i != 0:
                print( P + f"Time remaining = {estimate_time(ti,i+1,n)}" + W)
    
    else:      
        ref = gauss_map(np.shape(datacube)[-2], np.shape(datacube)[-1], np.max(datacube))
        datacube = align(datacube,ref)
    
    return datacube

def gauss_map(dx,dy,A):
    Y,X = np.indices((dx,dy))
    return Gauss2D([X,Y],A,5,dx/2,dy/2,0)
    
# def radius_map(dx,dy):
#     '''
#     Returns a 2D array of size (dx,dy) where each value is the distance to the center

#     Parameters
#     ----------
#     dx : int
#         horizontal size.
#     dy : int
#         vertical size.

#     Returns
#     -------
#     map : 2D array
#         The radius map.

#     '''
    
#     x,y = np.arange(0,dx), np.arange(0,dy)
#     X,Y = np.meshgrid(x,y)
    
#     X = X - dx/2
#     Y = Y - dy/2
    
#     return np.sqrt(X**2 + Y**2)

def Gauss2D(XY,A,fwhm,y0,x0,c):
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
    
    return A * np.exp(-((x-x0)**2 + (y-y0)**2)/(2*s**2)) + c
    
def Gauss2D_ravel(XY,A,fwhm,x0,y0,c):
    return np.ravel(Gauss2D(XY,A,fwhm,x0,y0,c))

def get_gaussian_to_image(img):
    '''
    Returns the parameters needed for Gauss2D, fitted from an image.

    Parameters
    ----------
    img : 2D array
        The image to fit the gaussian to.

    Returns
    -------
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

    '''
    guess = [np.max(img),10,0,0,0]
    
    y,x = np.indices(np.shape(img))
    x = x - np.shape(x)[1]/2
    y = y - np.shape(y)[0]/2
    
    try:
        popt, pcov = curve_fit(Gauss2D_ravel,(x,y),img.ravel(),p0=guess)
    except RuntimeError:
        print(R + "RunTimeError", " Sending back absurd values" + W)
        popt = [0,1000,0,0,0]
        pass
    
    return popt

def save_centered_to_file(path):
    '''
    Saves a centered .fits file of the datacube given by path.

    Parameters
    ----------
    path : str
        Path to the .fits file to center.
    '''
    # Checking if file exists and valid
    if not os.path.exists(path) or not ".fits" in path:
        return
    
    # Copy of datacube
    with fits.open(path) as hdul:
        img = hdul[0].data
        hdr = hdul[0].header
        
        # Replace properties in header
        hdr.set("ORIGFILE",hdr.get("ORIGFILE").replace("CLEAN","CENTERED"))
        hdr.set("HIERARCH ESO DPR TYPE",'CENTERED')
        
        # Replacing hdr, img
        print("\033[92m"+"Working on : " + path)
        img = align_datacube(img)
        print(G+ path.replace("clean", "centered") + " saved!\n" + W)
        
        n_hdul = fits.PrimaryHDU(img,header=hdr)
        n_hdul.writeto(path.replace("clean", "centered")
                               ,output_verify='silentfix',overwrite=True)
    return

def estimate_time(start, i, n):
    '''
    Estimates the time remaining after elapsed time (corresponding to i images)

    Parameters
    ----------
    start : nbr
        the starting time.
    i : int
        current frame.
    n : int
        total number of frames.

    Returns
    -------
    total : nbr
        Remaining time.

    '''
    elapsed = time.time() - start
    return int(elapsed * (n-i)/i)

def get_fwhm_from_datacube(datacube):
    '''
    Returns an array containing the fwmh from the curve fitted gaussian for every frame of the datacube.
    Datacube must be aligned first.

    Parameters
    ----------
    datacube : 3D array [time,x,y]
        The set of images to get fwhm from.

    Returns
    -------
    fwhm : array
        The value of each fwmh corresponding to every frame of the datacube.

    '''
    ti = time.time()
    
    if len(np.shape(datacube)) == 3:
        n = np.shape(datacube)[0]
        fwhm = []
        
        for i in range(n):
            fwhm.append(abs(get_gaussian_to_image(datacube[i,:,:])[1]))
            if i % 500 == 0 and i != 0:
                print(P + f"Time remaining = {estimate_time(ti,i+1,n)}" + W)
        
        return fwhm
    
    else:
        fwhm = []
        fwhm.append(abs(get_gaussian_to_image(datacube)[1]))
        return fwhm

def save_fwhm_to_file(path):
    '''
    Saves a .csv file representing a dataframe [frame in datacube, fwhm]

    Parameters
    ----------
    path : str
        The path to the given centered datacube

    '''
    # Checking if file exists and valid
    if not os.path.exists(path) or not ".fits" in path:
        return
    
    # Open export
    with fits.open(path) as hdul:
        datacube = hdul[0].data
    
    name = path[:-5].replace("centered", "fwhm")
    df = pd.DataFrame(get_fwhm_from_datacube(datacube))
    df.to_csv(name+".csv")
    return

def save_mean_image(fwhm,centered,percent=10):
    '''
    Saves to file the mean of the given datacube, after selecting the percent % best ones.

    Parameters
    ----------
    fwhm : str
        Path to the .csv containing the fwhm of each frame
    centered : str
        Path to the .fits datacube of each centered frame
    percent : int
        % of frames to keep.

    '''
    # Checking if file exists and valid
    if not os.path.exists(centered) or not ".fits" in centered or not os.path.exists(fwhm) or not ".csv" in fwhm:
        return
    
    # Open fwhm
    df = pd.read_csv(fwhm)
    df = df.sort_values(by="0")
    # Array of the indices of the best frames
    fwhm = df.index[1:2+int(min(100,percent)/100*len(df))]
    
    # Copy of datacube
    with fits.open(centered) as hdul:
        img = hdul[0].data
        hdr = hdul[0].header
        
        if len(np.shape(img)) == 3 and np.shape(img)[0] != 1:
            # Select and mean of 3D array
            img = img[fwhm]
            img = np.mean(img,axis=0)
            
        elif np.shape(img)[0] == 1:
            img = img[0,:,:]
        
        # Replace properties in header
        hdr.set("ORIGFILE",hdr.get("ORIGFILE").replace("CENTERED","MEAN"))
        hdr.set("HIERARCH ESO DPR TYPE",'MEAN')
    
        # Replacing hdr, img
        print("\033[92m"+"Working on : " + centered)
        print("\033[92m"+ centered.replace("centered", "mean") + " saved!")
        
        n_hdul = fits.PrimaryHDU(img,header=hdr)
        n_hdul.writeto(centered.replace("centered", "mean")
                               ,output_verify='silentfix',overwrite=True)
    return

def interpolate_bad_pixel(img,mask):
    '''
    Interpolates the pixels given by the mask array

    Parameters
    ----------
    img : 2D/3D array
        The image containing the bad pixels to remove.
    mask : 2D array
        Same size as img, 0 everywhere, 1 when the pixel needs to be corrected.

    Returns
    -------
    interpol : 2D/3D array
        Interpolated image
    '''
    if len(np.shape(img)) == 3:
        for i in range(np.shape(img)[0]):
            img[i,:,:] = interpolate_bad_pixel(img[i,:,:], mask)
        return img
            
            
    else:
        index = np.transpose(np.nonzero(img == 1))
        
        kernel = np.ones((3,3))
        kernel[1,1] = 0
        kernel /= 8
        
        conv = convolve2d(img,kernel,mode="same")
        
        img[index] = conv[index]
        
        return img      


main = "/home/tdewacher/Documents/Stage/" 
folders = ["P82-2008-2009","P88-2011-2012","P90-2012-2013","P94-2014-2015"]
# folders = ["P90-2012-2013"]

start = time.time()
clean_list = []

for f in folders:
    for obj in os.listdir(main+f):
        if obj !="Aldebaran" and obj != "Betelgeuse": 
            continue
        for times in os.listdir(main+f+"/"+obj):
            for filt in os.listdir(main+f+"/"+obj+"/"+times):
                if os.path.exists(main+f+"/"+obj+"/"+times+"/" + filt + "/export"):
                    for name in os.listdir(main+f+"/"+obj+"/"+times+"/" + filt + "/export"):
                        
                        if "_bad" in name:
                            continue
                        file = main+f+"/"+obj+"/"+times+"/" + filt + "/export/" + name +"/"+name
                        clean_list.append(file)


# # Manual Selecting good images
# for f in clean_list:
#     f = f+"_clean.fits"
#     with fits.open(f) as hdul:
#         if len(np.shape(hdul[0].data)) == 3:
#             img = hdul[0].data[0,:,:]
#         else:
#             img = hdul[0].data[:,:]
            
#         plt.figure()
#         img = np.nan_to_num(img)
#         plt.subplot(1,2,1)
#         name = f.split('/')[-1]
#         plt.imshow(img)
#         plt.title(name)
        
#         plt.subplot(1,2,2)
#         plt.hist(np.ravel(img),bins=50,log=True)
#         plt.title("Distribution")
        
#         plt.show()
        
#         keep = input("Keep? (Y/N)")
        
#         if keep == "N":
#             path = f[:-len(name)-1]
            
#             os.rename(path,path+"_bad")
            
#         plt.close('all')
            


# print(B + f"{len(clean_list)} files to work on" + W)

# for i in range(len(clean_list)):
#     f = clean_list[i]
#     if os.path.exists(f+"_mean.fits"):
#         continue
#     print(B + f"{i+1}/{len(clean_list)} \n" + W)
#     print(G + "Centering..." + W)
#     # save_centered_to_file(f+"_clean.fits")
#     print(G + "Getting FWHM..." + W)
#     # save_fwhm_to_file(f+"_centered.fits")
#     print(G + "Meaning..." + W)
#     save_mean_image(f+"_fwhm.csv", f+"_centered.fits")

# print(G + f"Temps écoulé : {int(time.time() - start)}s" + W)
