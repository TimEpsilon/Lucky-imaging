#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 11 17:17:22 2022

@author: tdewacher
"""

import numpy as np
import pandas as pd
import os
from astropy.io import fits
from astropy.visualization import wcsaxes
from astropy.wcs import WCS

def crop_image(img,dx):
    '''
    Crops the given image, centered on the middle, to the given dimension.
    If the image is a datacube (supposed of shape (T,Y,X)), the datacube will be cropped

    Parameters
    ----------
    img : 2D/3D array
        image to crop.
    dx : int
        width.

    Returns
    -------
    c_img : 2D/3D array
        The cropped image.

    '''
    Dy,Dx = np.shape(img)

    return img[int(Dy/2-dx/2):int(Dy/2+dx/2) , int(Dx/2-dx/2):int(Dx/2+dx/2)]

def get_deconvolution_datacube(directory):
    '''
    Returns a list containing the path to every deconvolution datacube

    Parameters
    ----------
    directory : str

    Returns
    -------
    list

    '''
    deconv = []
    
    for objet in os.listdir(directory):
        if "Betelgeuse" not in objet:
            continue
        for t in os.listdir(directory+objet):
            for filt in os.listdir(directory+objet+"/"+t):
                if not os.path.exists(directory+objet+"/"+t+"/"+filt+"/export") :
                    continue
                for file in os.listdir(directory+objet+"/"+t+"/"+filt+"/export"):
                    path = directory+objet+"/"+t+"/"+filt+"/export/"+file+"/"+file+"_deconvolution.fits"
                    if not os.path.exists(path):
                        continue
                    deconv.append(path)
                    
    return deconv

def create_selection_csv(directory):
    '''
    Creates a csv to complete with the image id (1-26) of every deconvolution datacube

    Parameters
    ----------
    directory : str
        path

    '''
    deconv = get_deconvolution_datacube(directory)
    i = np.zeros_like(deconv)
    
    # Cancel is exists
    if os.path.exists(directory+"/deconvolution_selection.csv"):
        return
    
    # Save csv
    df = pd.DataFrame(data={"Path":deconv,"ID":i})
    df.to_csv(directory+"/deconvolution_selection.csv")
    
    
def read_selection_csv(directory):
    '''
    Returns the completed dataframe

    Parameters
    ----------
    directory : str
        path

    Returns
    -------
    df : dataframe
        [PATH,ID]

    '''
    path = directory + "deconvolution_selection.csv"
    
    # Create if not exist
    if not os.path.exists(path):
        create_selection_csv(directory)
        return
    
    df = pd.read_csv(path)
    return df

def final_by_filter(directory):
    '''
    Creates a final best image for every filter

    Parameters
    ----------
    directory : str
        path

    '''
    df = read_selection_csv(directory)
    
    # Get every filter
    filters = []
    for path in df["Path"]:
        f = path.split('/')[-4].replace(",","")
        if not f in filters:
            filters.append(f)
            
    # Iterate over filters
    for f in filters:
        df_filt = df[df["Path"].str.contains(f)].reset_index()
        # Path to filter folder
        filt_path = df_filt["Path"][0]
        indices = [pos for pos,char in enumerate(filt_path) if char == "/"]
        filt_path = filt_path[:indices[8]]

        
        # Iterate over row at same filter
        for index,row in df_filt.iterrows():
            i = int(row["ID"])-1
            # Select 
            with fits.open(row["Path"]) as hdul:
                img = hdul[0].data[i,:,:]
                hdr = hdul[0].header
                hdr["NO_ITER"] = i*2
                
            # Save image
            hdul = fits.PrimaryHDU(img,header=hdr)
            hdul.writeto(row["Path"].replace("deconvolution","final"),overwrite=True)
            
        # Create selection file if not exist
        # In order to work, the user needs to delete the unwanted paths from the file
        # Only the first line will be read
        if not os.path.exists(filt_path+"/bestoff_filter.txt"):
            with open(filt_path+"/bestoff_filter.txt","w") as text:
                for path in df_filt["Path"]:
                    text.write(path+"\n")
                    
        # Getting the selected image
        with open(filt_path+"/bestoff_filter.txt","r") as text:
            path = text.readline()[:-1]
            
        # Opening the image
        with fits.open(path.replace("deconvolution","final")) as hdul:
            hdr = hdul[0].header
            img = hdul[0].data
            
        # Copying the image
        hdul = fits.PrimaryHDU(img,header=hdr)
        hdul.writeto(filt_path+"/bestoff_filter.fits",overwrite=True)
        
        
def createDatacubeOfEpoch(directory):
    '''
    Creates a datacube, sorted from low to high wavelengths, of every best image at a given epoch

    Parameters
    ----------
    directory : str
        path.

    '''
    # Create final images
    final_by_filter(directory)
    
    # Get every deconvolved image
    deconv = get_deconvolution_datacube(directory)
    bestoff = []
    wavelength = []
    
    # Get the best ones and their wavelengths
    for i,f in enumerate(deconv):
        indices = [pos for pos,char in enumerate(f) if char == "/"]
        path = f[:indices[8]]+"/bestoff_filter.fits"
        if os.path.exists(path) and not path in bestoff:
            bestoff.append(path)
            wavelength.append(float(f[indices[7]+4:indices[8]-1]))
            
    # Sorting by wavelength 
    index = range(len(wavelength))
    index = [x for _,x in sorted(zip(wavelength,index))]
    bestoff = [bestoff[i] for i in index]
    
    # Stacking
    imgs = []
    dx = 1000
    for path in bestoff:
        with fits.open(path) as hdul:
            img = hdul[0].data
            imgs.append(img)
            dx = min(np.shape(img)[0],dx)
    
    # Cropping
    for i,img in enumerate(imgs):
        imgs[i] = crop_image(img, dx)
        
    imgs = np.stack((imgs))
    hdul = fits.PrimaryHDU(imgs)
    
    
    
    hdul.writeto(directory+"Betel_AllFilters.fits")

# =============================================================================
#                                  MAIN
# =============================================================================

directory = "/home/tdewacher/Documents/Stage/P82-2008-2009/"
# createDatacubeOfEpoch(directory)
