# -*- coding: utf-8 -*-
"""
Created on Wed Jun  8 09:50:12 2022

@author: timde
"""
import numpy as np
import pandas as pd
import os
from astropy.io import fits


def image_median(directory):
    """
    Returns the median of every pixel of a given dataset in a directory.
    
    Parameters
    ---------
    directory: str
        Path to the dataset
    
    Returns
    --------
    
    image_median: 2D array
        The per-pixel median image
    
    """
    median = np.array([])
    
    img_list = []
    for f in os.listdir(directory):
        img_list = []
        input_file = os.path.join(directory, f)
        if ".fits" not in f: continue
        
        with fits.open(input_file) as hdul1:
            img = hdul1[0].data
            # Excluding 3D arrays
            if len(np.shape(img)) !=2: continue
            # List of images
            img_list.append(img)
            
    # 3D array (image stacking)
    if len(img_list) == 0: return
    median = np.dstack((img_list))
    return np.median(median,axis=2)

def save_median_to_file(directory):
    """
    Saves the median file in a new folder in directory.
    
    Parameters
    -------
    directory: str
        Path to the folder
    """
    if len(os.listdir(directory)) == 0: return
    
    # Getting the image
    img = image_median(directory)
    if img is None: return
    
    # Create an export folder if not exist
    exp_path = os.path.join(directory,"master")
    if not os.path.exists(exp_path):
        os.makedirs(exp_path)
        print(exp_path + " created!")
        
    # Copy of first fits file found
    for f in os.listdir(directory):
        if ".fits" in f: 
            input_file = os.path.join(directory, f)
            with fits.open(input_file) as hdul:
                hdul.writeto(exp_path + "/median.fits",output_verify='silentfix',overwrite=True)
            break
        
    median_file = os.path.join(exp_path,"median.fits")
    with fits.open(median_file,mode="update") as hdul:
        # Set median image as data
        hdul[0].data = img
        hdr = hdul[0].header
        
        # Replace properties in header
        hdr.set("ORIGFILE","MEDIAN_"+ "-" +directory.split("/")[-2].replace(',','').upper()
                + "-" +directory.split("/")[-1].upper()+".fits")
        
        hdul[0].header = hdr

def save_master_dark_to_file(directory):
    """
    Saves the master dark in the given directory.
    The median dark must be contained in directory.

    Parameters
    ----------
    directory : str
        Path to the folder.

    """
    # Copy of median
    for f in os.listdir(directory):
        if "median" in f: 
            input_file = os.path.join(directory, f)
            with fits.open(input_file) as hdul:
                hdul.writeto(directory + "/master.fits",output_verify='silentfix',overwrite=True)
            break
        
    master = os.path.join(directory,"master.fits")
    with fits.open(master,mode="update") as hdul:
        # Set median image as data
        hdr = hdul[0].header
        
        # Replace properties in header
        hdr.set("ORIGFILE","MASTER_"+ "-" +directory.split("/")[-3].replace(',','').upper()
                + "-" +directory.split("/")[-2].upper()+".fits")
        
        hdul[0].header = hdr

def get_corresponding_dark(directory):
    """
    Returns the string path to the master dark of same integration time.
    Returns None if file doesn't exist.

    Parameters
    ----------
    directory : str
        The path to the directory containing the median flat.

    Returns
    -------
    dark : str
        The path to the master dark of same time.

    """
    time = directory.split('/')[-2]
    flat = directory.split('/')[-3]
    
    directory= directory.removesuffix(flat+"/"+time+"/master")
    directory += "DARK/" + time + "/master/master.fits"
    
    path = os.path.join(directory)
    if not os.path.exists(path): 
        return
    else:
        return directory
    
def get_corresponding_masters(directory):
    """
    Returns the string paths to the master dark/flat of same integration time.
    If no flat is found, another master flat will be used.
    Returns None if files don't exist.

    Parameters
    ----------
    directory : str
        The path to the datacube.

    Returns
    -------
    dark,flat : str
        The paths to the master dark and flat of same time.

    """
    time = directory.split('/')[-2]
    star = directory.split('/')[-3]
    
    # Dir names
    directory = directory.removesuffix(star+"/"+time+"/"+directory.split('/')[-1])
    dark = directory + "DARK/" + time + "/master/master.fits"
    flat = directory + "FLAT,SKY/" + time + "/master/master.fits"
    
    # If paths exist
    if os.path.exists(dark) and os.path.exists(flat): 
        return dark,flat
    elif os.path.exists(dark):
        for f in os.listdir(directory + "FLAT,SKY"):
            flat = directory + "FLAT,SKY/" + f + "/master/master.fits"
            if os.path.exists(flat): 
                return dark,flat
    return None, None
   
def save_master_flat_to_file(directory):
    """
    Saves the master flat in the given directory.
    The median flat must be contained in directory.
    If no corresponding dark .fits file exists, the operation is cancelled.

    Parameters
    ----------
    directory : str
        Path to the folder.

    """
    # Copy of median
    for f in os.listdir(directory):
        if "median" in f: 
            input_file = os.path.join(directory, f)
            with fits.open(input_file) as hdul:
                hdul.writeto(directory + "/master.fits",output_verify='silentfix',overwrite=True)
            break
    
    # Get flat path
    flat = os.path.join(directory,"master.fits")
    
    # Get dark path
    dark = get_corresponding_dark(directory)
    if dark is None:
        return
    
    
    # Opens master flat
    with fits.open(flat,mode="update") as hdul:
        flat_img = hdul[0].data
        hdr = hdul[0].header
        
        # Opens master dark
        with fits.open(dark) as hdul1:
            dark_img = hdul1[0].data
            
            # Master flat image
            master_flat = flat_img - dark_img
            master_flat /= np.median(master_flat)
            
            hdul[0].data = master_flat
        
        # Replace properties in header
        hdr.set("ORIGFILE","MASTER_"+ "-" +directory.split("/")[-3].replace(',','').upper()
                + "-" +directory.split("/")[-2].upper()+".fits")
        
        hdul[0].header = hdr  


def save_clean_images(file):
    """
    Saves the cleaned datacube after correcting biases with a master dark and a master flat.
    The median flat must be contained in directory.
    If no corresponding dark .fits file exists, the operation is cancelled.

    Parameters
    ----------
    file : str
        Path to the datacube.

    """
    dark,flat = get_corresponding_masters(file)
    
    # If files not found, cancel
    if dark is None or flat is None:
        return
    
    # Create an export folder if not exist
    exp_path = file.removesuffix(file.split('/')[-1]) + "export" 
    if not os.path.exists(exp_path):
        os.makedirs(exp_path)
        print(exp_path + " created!")
    
    
    # Copy of datacube
    with fits.open(file) as hdul_l:
        hdul_l.writeto(exp_path + "/" + file.split('/')[-1]
                               ,output_verify='silentfix',overwrite=True)
    
    # Open export
    with fits.open(exp_path+"/"+file.split('/')[-1]) as hdul_l:
        img_l = hdul_l[0].data
        hdr = hdul_l[0].header
        
        # Replace properties in header
        hdr.set("ORIGFILE","UNBIASED_"+ "-" +file.split("/")[-3].replace(',','').upper()
                + "-" +file.split("/")[-2].upper()+".fits")
        
        # Getting master dark and flat images
        with fits.open(dark) as hdul_d:
            img_d = hdul_d[0].data
        with fits.open(flat) as hdul_f:
            img_f = hdul_f[0].data
    
        # For every frame of the datacube
        for i in range(np.shape(img_l)[0]):
            img_l[i,:,:] = (img_l[i,:,:] - img_d)/img_f
            
        hdul_l[0].header = hdr
        hdul_l[0].data = img_l