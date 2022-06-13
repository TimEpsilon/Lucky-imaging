# -*- coding: utf-8 -*-
"""
Created on Wed Jun  8 09:50:12 2022

@author: timde
"""
import numpy as np
import pandas as pd
import os
from astropy.io import fits
import matplotlib.pyplot as plt
import datetime


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
            # Only 1s element of 3D arrays
            if len(np.shape(img)) == 3: img = img[0,:,:]
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
    # Ignoring empty folders and the export folder
    if len(os.listdir(directory)) == 0: return
    if "master" in directory: return
    
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
        directory = directory.replace('\\','/')
        hdr.set("ORIGFILE","MEDIAN_" +directory.split("/")[-2].replace(',','').upper()
                + "-" +directory.split("/")[-1].upper()+".fits")
        
        hdul[0].header = hdr

def save_master_dark_to_file(directory):
    """
    Saves the master dark in the given directory.
    The median dark must be contained in directory, else, does nothing.

    Parameters
    ----------
    directory : str
        Path to the folder.

    """
    # Copy of median
    for f in os.listdir(directory):
        if "median.fits" in f: 
            input_file = os.path.join(directory, f)
            with fits.open(input_file) as hdul:
                hdul.writeto(directory + "/master.fits",output_verify='silentfix',overwrite=True)
            break
        
    master = os.path.join(directory,"master.fits")
    if not os.path.exists(master): return
    
    with fits.open(master,mode="update") as hdul:
        # Set median image as data
        hdr = hdul[0].header
        
        # Replace properties in header
        directory = directory.replace('\\','/')
        hdr.set("ORIGFILE","MASTER_" +directory.split("/")[-4].replace(',','').upper()
                + "-" +directory.split("/")[-3].upper()+".fits")
        
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
    directory = directory.replace('\\','/')
    filters = directory.split('/')[-2]
    time = directory.split('/')[-3]
    flat = directory.split('/')[-4]
    
    directory= directory.removesuffix(flat+"/"+time+"/"+filters+"/master")
    directory += "DARK/" + time + "/closed,NB_2.17,NB_1.08,/master/master.fits"
    
    path = os.path.join(directory)
    if not os.path.exists(path): 
        return
    else:
        return directory
    
def get_corresponding_masters(directory):
    """
    Returns the string paths to the master dark/flat of same integration time/filter.
    Returns None if any file doesn't exist.

    Parameters
    ----------
    directory : str
        The path to the datacube.

    Returns
    -------
    dark,flat : str
        The paths to the master dark and flat of same time.

    """
    # Getting info
    directory = directory.replace('\\', '/')
    filters = directory.split('/')[-2]
    time = directory.split('/')[-3]
    star = directory.split('/')[-4]
    
    # Dir names
    directory = directory.removesuffix(star+"/"+time+"/"+filters+"/"+directory.split('/')[-1])
    dark = directory + "DARK/" + time + "/closed,NB_2.17,NB_1.08,/master/master.fits"
    # Checking in other folders for same filter
    t_flat = directory + "FLAT,SKY"
    if not os.path.exists(t_flat): return None,None
    for time in os.listdir(t_flat):
        flat = directory + "FLAT,SKY/" + time + "/" + filters + "/master/master.fits"
        if os.path.exists(flat):
            break
    
        
    # If paths exist
    if os.path.exists(dark) and os.path.exists(flat): 
        return dark,flat
    elif os.path.exists(dark):
        return dark,None
    elif os.path.exists(flat):
        return None,flat
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
        if "median.fits" in f: 
            input_file = os.path.join(directory, f)
            with fits.open(input_file) as hdul:
                hdul.writeto(directory + "/master.fits",output_verify='silentfix',overwrite=True)
            break
    
    # Get flat path
    flat = os.path.join(directory,"master.fits")
    if not os.path.exists(flat): return
    
    # Get dark path
    dark = get_corresponding_dark(directory)
    if dark is None: return
    
    
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


def save_clean_images(file,text_file):
    """
    Saves the cleaned datacube after correcting biases with a master dark and a master flat.
    If no corresponding dark and flat .fits files exist, the operation is cancelled.
    The dark must be of same integration time, the flat of same filter.

    Parameters
    ----------
    file : str
        Path to the datacube.

    """
    file = file.replace("\\","/")
    dark,flat = get_corresponding_masters(file)
    
    #######
    filters = file.split('/')[-2]
    time = file.split('/')[-3]
    text_file.write(file[33:]+";"+filters+";"+time+";"+str(not dark is None)+";"+str(not flat is None)+"\n")
    #######
    
    # If files not found, cancel
    if dark is None or flat is None:
        return
    
    
    name = file.split('/')[-1][:-5]
    # Create an export folder if not exist
    exp_path = file.removesuffix(file.split('/')[-1]) + "export" 
    if not os.path.exists(exp_path):
        os.makedirs(exp_path)
        print(exp_path + " created!")
    glob_exp_path = exp_path + "/" + name
    if not os.path.exists(glob_exp_path):
        os.makedirs(glob_exp_path)
        print(glob_exp_path + " created!")
    
    
    # Copy of datacube
    with fits.open(file) as hdul_l:
        hdul_l.writeto(glob_exp_path + "/" + file.split('/')[-1][:-5] + "_clean.fits"
                               ,output_verify='silentfix',overwrite=True)
    
    # Open export
    with fits.open(glob_exp_path + "/" + file.split('/')[-1][:-5] + "_clean.fits",mode='update') as hdul_l:
        img_l = hdul_l[0].data
        hdr = hdul_l[0].header
        
        # Replace properties in header
        hdr.set("ORIGFILE","CLEAN_" +file.split("/")[-4].upper()
                + "-" +file.split("/")[-3].upper()
                + "-" +file.split("/")[-2].upper()
                +".fits")
        
        hdr.set("HIERARCH ESO DPR TYPE",'CLEAN')
        hdr.set("HISTORY", datetime.date.today().ctime().replace(':','-') + " - Created file")
        
        # Getting master dark and flat images
        with fits.open(dark) as hdul_d:
            img_d = hdul_d[0].data
        with fits.open(flat) as hdul_f:
            img_f = hdul_f[0].data
    
        # Cropping images
        dx,dy = np.shape(img_d)
        Dx,Dy = np.shape(img_l[0,:,:])
        c_img_d = img_d[int(dx/2-Dx/2):int(dx/2+Dx/2+1) , int(dy/2-Dy/2):int(dy/2+Dy/2+1)]
        dx,dy = np.shape(img_f)
        c_img_f = img_f[int(dx/2-Dx/2):int(dx/2+Dx/2) , int(dy/2-Dy/2):int(dy/2+Dy/2)]
    
        # For every frame of the datacube
        c_img_d = np.repeat(c_img_d[np.newaxis,:,:],np.shape(img_l)[0],axis=0)
        c_img_f = np.repeat(c_img_f[np.newaxis,:,:],np.shape(img_l)[0],axis=0)
        
        img_l = (img_l - c_img_d)/c_img_f
        
        # plt.figure()
        # plt.imshow(c_img_d[0,:,:])
        # plt.figure()
        # plt.imshow(c_img_f[0,:,:])
        # plt.figure()
        # plt.imshow(img_l[0,:,:])
        # plt.figure()
        # plt.imshow(hdul_l[0].data[0])
        # plt.show()
            
        hdul_l[0].header = hdr
        hdul_l[0].data = img_l
        

def iterate_over_tree(main, key, method):
    """
    Applies the method on every folder named after key or inside, starting from main
    
    Parameters
    ----------
    main: str
        Path to the starting folder
        
    key: str
        Which folder should the method be applied to
    
    method: func
        A function that applies on a given folder path
    
    """
    
    for f in os.listdir(main):
        path = os.path.join(main,f)
        
        if os.path.isdir(path):
            if key in path:
                method(path)
            # Recursive call when dir encountered
            iterate_over_tree(path, key, method)
     
        
def clean_all(path,text_file):
    '''
    Cleans every possible .fits file in a given star directory

    Parameters
    ----------
    path : str
        Path to the star directory.
    '''
    for time in os.listdir(path):
        for filters in os.listdir(path+"/"+time):
            for file in os.listdir(path+"/"+time+"/"+filters):
                fpath = os.path.join(path,time,filters,file)
                if ".fits" in fpath:
                    save_clean_images(fpath,text_file)
    

main = "/home/tdewacher/Documents/Stage/" 
folders = ["P82-2008-2009","P88-2011-2012","P90-2012-2013","P94-2014-2015"]
master = ["DARK","FLAT,SKY"]

with open("cleaned_Output.csv", "w") as text_file:
    text_file.write("Path;Filter;Time;Dark;Flat\n")
    for f in folders:
        # for m in master:
        # iterate_over_tree(main+f,m,save_median_to_file)
        # iterate_over_tree(main+f,"DARK",save_master_dark_to_file)
        # iterate_over_tree(main+f,"SKY",save_master_dark_to_file)
        # iterate_over_tree(main+f,"FLAT,SKY",save_master_flat_to_file)
        clean_all(main+f+"/Betelgeuse",text_file)
        clean_all(main+f+"/Aldebaran",text_file)
    
