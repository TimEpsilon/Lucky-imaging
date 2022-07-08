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
from matplotlib.colors import PowerNorm


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
    dx = 10000000
    dy = 10000000
    for f in os.listdir(directory):
        input_file = os.path.join(directory, f)
        if ".fits" not in f: continue
        
        with fits.open(input_file) as hdul1:
            img = hdul1[0].data
            # append each element of 3D arrays
            if len(np.shape(img)) == 3: 
                dx = min(dx,np.shape(img[0,:,:])[1])
                dy = min(dy,np.shape(img[0,:,:])[0])
                for i in range(np.shape(img)[0]):
                    img_list.append(img[i,:,:])
                continue
            # List of images
            dx = min(dx,np.shape(img)[1])
            dy = min(dy,np.shape(img)[0])
            img_list.append(img)
            
    # 3D array (image stacking)
    if len(img_list) == 0: return
    
    for i in range(len(img_list)):
        img_list[i] = crop_image(img_list[i], dx, dy)
        
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
                # Set median image as data
                hdr = hdul[0].header
                
                # Replace properties in header
                directory = directory.replace('\\','/')
                hdr.set("ORIGFILE","MEDIAN_" +directory.split("/")[-2].replace(',','').upper()
                        + "-" +directory.split("/")[-1].upper()+".fits")
                
                # Make copy
                n_hdul = fits.PrimaryHDU(img,header=hdr)
                n_hdul.writeto(exp_path + "/median.fits",output_verify='silentfix',overwrite=True)
            break

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
                img = hdul[0].data
                hdr = hdul[0].header
                
                # Replace properties in header
                directory = directory.replace('\\','/')
                hdr.set("ORIGFILE","MASTER_" +directory.split("/")[-4].replace(',','').upper()
                        + "-" +directory.split("/")[-3].upper()+".fits")
                
                n_hdul = fits.PrimaryHDU(img,header=hdr)
                n_hdul.writeto(directory + "/master.fits",output_verify='silentfix',overwrite=True)
            return

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
    
def get_corresponding_masters(directory,flattype):
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
    t_flat = directory + flattype
    if not os.path.exists(t_flat): return None,None
    for time in os.listdir(t_flat):
        flat = directory + flattype + time + "/" + filters + "/master/master.fits"
        if os.path.exists(flat):
            break
    
        
    # If paths exist
    if os.path.exists(dark) and os.path.exists(flat): 
        return dark,flat
    elif os.path.exists(dark):
        
        if os.path.exists(flat):
            return dark,flat
        else:
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
                img = hdul[0].data
                hdr = hdul[0].header
                
                # Get flat path
                flat = os.path.join(directory,"master.fits")
                # if not os.path.exists(flat): return
                
                # Get dark path
                dark = get_corresponding_dark(directory)
                if dark is None: return
                
                # Opens master dark
                with fits.open(dark) as hdul1:
                    dark_img = hdul1[0].data
                    
                dx = min(np.shape(img)[-1],np.shape(dark_img)[-1])
                dy = min(np.shape(img)[-2],np.shape(dark_img)[-2])
                    
                img = crop_image(img, dx, dy)
                dark_img = crop_image(dark_img, dx, dy)
                
                # Master flat image
                img = img - dark_img
                img /= np.median(img)
                
                
                # Replace properties in header
                hdr.set("ORIGFILE","MASTER_"+ "-" +directory.split("/")[-3].replace(',','').upper()
                        + "-" +directory.split("/")[-2].upper()+".fits")
                
                n_hdul = fits.PrimaryHDU(img,header=hdr)
                n_hdul.writeto(directory + "/master.fits",output_verify='silentfix',overwrite=True)
            return


def save_clean_images(file,text_file=None):
    """
    Saves the calibrated datacube after correcting biases with a master dark and a master flat.
    If no corresponding dark and flat .fits files exist, the operation is cancelled.
    The dark must be of same integration time, the flat of same filter.

    Parameters
    ----------
    file : str
        Path to the datacube.

    """
    file = file.replace("\\","/")
    dark,flat = get_corresponding_masters(file,"FLAT,SKY/")
    
    if flat is None:
        dark,flat = get_corresponding_masters(file,"FLAT,LAMP/")
    
    #######
    if text_file is not None:
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
        img_l = hdul_l[0].data
        hdr = hdul_l[0].header
        
        # Replace properties in header
        hdr.set("ORIGFILE","CLEAN" +file.split("/")[-4].upper()
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

        dx = min(np.shape(img_d)[-1],np.shape(img_f)[-1],np.shape(img_l)[-1])
        dy = min(np.shape(img_d)[-2],np.shape(img_f)[-2],np.shape(img_l)[-2])
        
        # Cropping images
        img_d = crop_image(img_d, dx, dy)
        img_l = crop_image(img_l, dx, dy)
        img_f = crop_image(img_f, dx, dy)
        
        # For every frame of the datacube
        if len(np.shape(img_l)) == 3:
            img_d_3d = np.repeat(img_d[np.newaxis,:,:],np.shape(img_l)[0],axis=0)
            img_f_3d = np.repeat(img_f[np.newaxis,:,:],np.shape(img_l)[0],axis=0)
        else :
            img_d_3d = img_d
            img_f_3d = img_f
        
        img_l = (img_l - img_d_3d)
        
        img_l[img_f_3d>0.05] /= img_f_3d[img_f_3d>0.05]
        
        # Save copy
        n_hdul_l = fits.PrimaryHDU(img_l,header=hdr)
        n_hdul_l.writeto(glob_exp_path + "/" + file.split('/')[-1][:-5] + "_clean.fits"
                               ,output_verify='silentfix',overwrite=True)
    
        

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
            if key in path.split("/"[-1]):
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
                    
                    
def remove_bad_pixel(img,Y,X):
    for i in range(len(X)):
        x,y = X[i],Y[i]
        if len((np.shape(img)))==3:            
            img[:,y,x] = (img[:,y+1,x] + img[:,y+1,x+1] + img[:,y,x+1] + img[:,y-1,x+1] + img[:,y-1,x] + img[:,y-1,x-1] + img[:,y,x-1] + img[:,y+1,x-1]) / 8 
        else:
            img[y,x] = (img[y+1,x] + img[y+1,x+1] + img[y,x+1] + img[y-1,x+1] + img[y-1,x] + img[y-1,x-1] + img[y,x-1] + img[y+1,x-1]) / 8 
    return img

def bad_pixel_to_file(f):
    paths = []

    for obj in os.listdir(main+f):
        if obj != "Betelgeuse" and obj != "Aldebaran": 
            continue
        for times in os.listdir(main+f+"/"+obj):
            for filt in os.listdir(main+f+"/"+obj+"/"+times):
                if os.path.exists(main+f+"/"+obj+"/"+times+"/" + filt + "/export"):
                    for name in os.listdir(main+f+"/"+obj+"/"+times+"/" + filt + "/export"):
                        if "_bad" in name:
                            continue
                        file = main+f+"/"+obj+"/"+times+"/" + filt + "/export/" + name +"/"+name
                        paths.append(file)
                        
    for path in paths:
        with fits.open(path+"_clean.fits") as hdul:
            img = hdul[0].data
            
            plt.figure()
            plt.subplot(1,2,1)
            plt.imshow(img[0,:,:],norm=PowerNorm(gamma=0.3))
            
            good = np.copy(img)
            
            
            #P90
            # good = remove_bad_pixel(good, [50,38,37,31,30,19,15,9,48,47,47,48], [36,46,46,36,36,24,37,33,51,50,51,50])
            
            #P94
            X = [65,23,22,22,90,90,62,60,66,70,70,76,71]
            Y = [113,52,52,53,12,13,68,72,79,87,69,61,68]
            good = remove_bad_pixel(good, Y, X)
            mask = np.zeros_like(img[0,:,:])
            for i in range (len(X)):
                mask[Y[i],X[i]] = 1
            
            
            plt.subplot(1,2,2)
            plt.imshow(mask)
            
            plt.suptitle(path.split("/")[-1])
            plt.show()
            hdul1 = fits.PrimaryHDU(good,header=hdul[0].header)
        hdul1.writeto(path+"_clean.fits",output_verify='silentfix',overwrite=True)
        
        
        
        
main = "/home/tdewacher/Documents/Stage/" 
# folders = ["P82-2008-2009","P88-2011-2012","P90-2012-2013","P94-2014-2015"]
folders = ["P94-2014-2015"]
master = ["DARK","FLAT,SKY","FLAT,LAMP"]

with open("cleaned_Output.csv", "w") as text_file:
#     text_file.write("Path;Filter;Time;Dark;Flat\n")
    for f in folders:
#         for m in master:
#             iterate_over_tree(main+f,m,save_median_to_file)
#         iterate_over_tree(main+f,"DARK",save_master_dark_to_file)
#         iterate_over_tree(main+f,"FLAT,SKY",save_master_flat_to_file)
#         iterate_over_tree(main+f,"FLAT,LAMP",save_master_flat_to_file)
        clean_all(main+f+"/Betelgeuse",text_file)
        clean_all(main+f+"/Aldebaran",text_file)

bad_pixel_to_file("P94-2014-2015")