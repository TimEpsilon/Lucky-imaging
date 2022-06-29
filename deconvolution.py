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
    return np.exp(-((X-dx/2)**2 + (Y-dy/2)**2)/(2*s**2))

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


def save_normalized_copy(psf,n_img):
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
    bckgr = 

    '''
    # Copy of psf
    with fits.open(psf) as hdul:
        img = hdul[0].data
        hdr = hdul[0].header
        
        dx = min(np.shape(n_img)[-1],np.shape(img)[-1])
        dy = min(np.shape(n_img)[-2],np.shape(img)[-2])
        
        img = crop_image(img, dx, dy)
        n_img = crop_image(n_img, dx, dy)
        
        img[img<0]=np.min(img[img>0])
        img[n_img<=0]=np.min(img[img>0])
        
        # Apodizing
        x,y = np.shape(hdul[0].data)
        gauss = gauss_map(x,y,20)
        img *= gauss
        img /= hdul[0].data.sum()
        
        img = np.nan_to_num(img)
        
        n_hdul = fits.PrimaryHDU(img,header=hdr)
        n_hdul.writeto(psf.replace("mean","norm")
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
        
    dx = min(np.shape(n_img)[-1],np.shape(img)[-1])
    dy = min(np.shape(n_img)[-2],np.shape(img)[-2])
    
    img = crop_image(img, dx, dy)
        
    img_min=0
    img[img<0] = np.min(img[img>0])
    
    n_hdul = fits.PrimaryHDU(img,header=hdr)
    n_hdul.writeto(path.replace("mean","norm")
                           ,output_verify='silentfix',overwrite=True)
    print(G + "Copy of " + path + W)      
    return img_min


def deconvolution(img,psf,n=15,img_min=0):
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

    if os.path.exists(img.replace("norm","deconvolution")):
        os.remove(img.replace("norm","deconvolution"))
    stsdas.analysis.restore.lucy(img, psf, img.replace("norm","deconvolution"), 40, 0,
                                 niter=n, limchisq=1e-10,verbose=False,nsave=5)
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
            return corr_path
    
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
    img_min = save_pos_copy(path,psf_path)
    
    # Getting img
    with fits.open(path.replace("mean","norm")) as img:
        d_img = img[0].data
        
    
    # Normalize psf copy
    save_normalized_copy(psf_path,d_img)
    psf_path.replace("mean","norm")
    
    with fits.open(psf_path.replace("mean","norm")) as psf:
        d_psf = psf[0].data
        
    # Saving deconvolution
    deconvolution(path.replace("mean","norm"),psf_path.replace("mean","norm"),img_min=img_min)
    
    # Getting deconv
    with fits.open(path.replace("mean", "deconvolution"),ignore_missing_end=True) as hdul:
        deconv = hdul[0].data
        hdr = hdul[0].header
        
    deconv += img_min
    
    hdul = fits.PrimaryHDU(deconv,header=hdr)
    hdul.writeto(path.replace("mean", "deconvolution"),overwrite=True,output_verify="silentfix")
    
    reconv = convolve2d(deconv, d_psf,mode='same')
    
    fig,ax = plt.subplots(2,2)
    
    ax[0,0].imshow(d_img,norm=PowerNorm(0.3),cmap="afmhot",interpolation='bicubic')
    ax[0,0].set_title("Betelgeuse : " + path.split('/')[-2])
    ax[1,1].imshow(d_psf,norm=PowerNorm(0.3),cmap="afmhot",interpolation='bicubic')
    ax[1,1].set_title("Aldebaran " + path.split('/')[-4])
    ax[1,0].imshow(deconv,norm=PowerNorm(0.3),cmap="afmhot",interpolation='bicubic')
    ax[1,0].set_title("Deconvolution")
    ax[0,1].imshow(reconv,norm=PowerNorm(0.3),cmap="afmhot",interpolation='bicubic')
    ax[0,1].set_title("Reconvolution")
    
    i = 0
    for f in os.listdir("/home/tdewacher/Images/"):
        if path.split("/")[-7] + "-" + path.split('/')[-4] + "_" + path.split("/")[-6] in f:
            i += 1
    
    plt.savefig("/home/tdewacher/Images/"+ path.split("/")[-7] + "-" + path.split('/')[-4] + "_" + path.split("/")[-6] + "-" + str(i) + ".png")
    return 

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

def getFwhm(img):
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
    
    return popt[1]

def getFwhmAndIteration(directory):
    i = []
    fwhm = []
    for f in os.listdir(directory):
        if "deconvolution" in f and ".fits" in f:
            with fits.open(directory+f) as hdul:
                img = hdul[0].data
                hdr = hdul[0].header
                
                i.append(int(hdr["NO_ITER"]))
                fwhm.append(getFwhm(img))
                
    return i,fwhm
        

def showPlanche(folders):
    
    start = time.time()
    df = pd.DataFrame(data={"path":[],"epoch":[],"filter":[],"type":[]})
    
    
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
                            df = df.append({"path":file,"epoch":f,"filter":filt,"type":obj},ignore_index=True)
                     
                            
    print(B + str(len(df)) + " files to work on" + W)
    
    for f in folders:
        epoch = df[df["epoch"] == f]
        filters = np.unique(epoch["filter"])
        n = len(filters)
        if n == 0: continue
        
        fig1,ax1 = plt.subplots(3,n)
        
        
        
        for i in range(n):
            filt = filters[i]
            
            files = epoch[epoch["filter"] == filt]
            betelgeuse = files[files["type"] == "Betelgeuse"]
            
            m = len(betelgeuse)
            
            plt.ion()
            fig,ax = plt.subplots(3,m)
            
            for j in range(m):
                with fits.open(betelgeuse["path"].to_numpy()[j]+"_mean.fits") as hdul:
                    ax[0,j].imshow(hdul[0].data,norm=PowerNorm(0.3),cmap="afmhot")
                    ax[0,j].set_title(betelgeuse["path"].to_numpy()[j].split('/')[-1][18:])      
            plt.suptitle(f + " : " + filt)
            plt.show(block=False)
            
            
            j = int(input("Betelgeuse id : "))
            
            with fits.open(betelgeuse["path"].to_numpy()[j]+"_mean.fits") as hdul:
                ax1[0,i].imshow(hdul[0].data,norm=PowerNorm(0.3),cmap="afmhot",interpolation="bicubic")
                ax1[0,i].set_title("Betelg - " + filt)
                
            apply_deconvolution(betelgeuse["path"].to_numpy()[j]+"_mean.fits")
            
            psf = get_matching_psf(betelgeuse["path"].to_numpy()[j]+"_mean.fits")
            
            if os.path.exists(psf.replace("mean","norm")):
                with fits.open(psf.replace("mean","norm")) as hdul:
                    ax1[2,i].imshow(hdul[0].data,norm=PowerNorm(0.3),cmap="afmhot",interpolation="bicubic")
                    ax1[2,i].set_title("Aldeb - " + filt)
                
            if os.path.exists(betelgeuse["path"].to_numpy()[j]+"_deconvolution.fits"):
                with fits.open(betelgeuse["path"].to_numpy()[j]+"_deconvolution.fits") as hdul:
                    ax1[1,i].imshow(hdul[0].data,norm=PowerNorm(0.3),cmap="afmhot",interpolation="bicubic")
                    ax1[1,i].set_title("Deconvol - " + filt)
                    
        
        plt.show()
    


main = "/home/tdewacher/Documents/Stage/" 
folders = ["P82-2008-2009","P88-2011-2012","P90-2012-2013","P94-2014-2015"]


showPlanche(folders)





# # folders = ["P82-2008-2009"]

# start = time.time()
# clean_list = []

# for f in folders:
#     for obj in os.listdir(main+f):
#         if obj != "Betelgeuse": 
#             continue
#         for times in os.listdir(main+f+"/"+obj):
#             for filt in os.listdir(main+f+"/"+obj+"/"+times):
#                 if os.path.exists(main+f+"/"+obj+"/"+times+"/" + filt + "/export"):
#                     for name in os.listdir(main+f+"/"+obj+"/"+times+"/" + filt + "/export"):
#                         if "_bad" in name:
#                             continue
#                         file = main+f+"/"+obj+"/"+times+"/" + filt + "/export/" + name +"/"+name
#                         clean_list.append(file)

# print(B + str(len(clean_list)) + " files to work on" + W)


# # for i in range(len(clean_list)):
# #     f = clean_list[i]
    
# #     print(G + "Working on " + f + W)
    
# #     filt = f.split("/")[-4]
# #     name = f.split("/")[-1]
    
# #     # if os.path.exists(f+"_deconvolution.fits"): continue
    
# #     # Deconvolution
# #     apply_deconvolution(f+"_mean.fits")
# #     print(B + str(i+1) + "/" + str(len(clean_list)) + " \n" + W)
    
    
# #     # Plot every image
# #     # filt = f.split('/')[-4]
# #     # epoch = f.split('/')[-7]
# #     # with fits.open(f+"_deconvolution.fits") as hdul:
# #     #     hdr = hdul[0].header
# #     #     img = hdul[0].data
        
# #     #     fov = hdr['CD2_2']*hdr['NAXIS2']/2
# #     #     fov *= 3600*1000
        
# #     #     plt.figure()
# #     #     plt.imshow(img,norm=PowerNorm(0.3),cmap="afmhot",origin="lower",
# #     #                 extent=[fov, -fov, -fov, fov], aspect="equal",interpolation='bicubic')
# #     #     plt.colorbar()
# #     #     plt.xlabel("Ascension droite relative (mas)")
# #     #     plt.ylabel("Declinaison relative (mas)")
# #     #     plt.title(epoch + " : " + filt)
        
# #     #     # Angular diameter
# #     #     star = plt.Circle((-fov/np.shape(img)[0],fov/np.shape(img)[1]), 43.7,color='b',fill=False,alpha=0.5)
# #     #     ax = plt.gca()
# #     #     ax.add_patch(star)
        
# #     #     plt.draw()
    

# # print(G + "Temps écoulé : " + str(int(time.time() - start)) + "s" + W)

# # plt.legend()
# # plt.show(block=False)

# fig,ax = plt.subplots(1,6)
# i = 0
# for sigma in [1,5,10,15,20,30]:
#     apply_deconvolution("/home/tdewacher/Documents/Stage/P82-2008-2009/Betelgeuse/0.007204/NB_1.26,/export/P82-2008-2009NACO.2009-01-03T02:23:51.357/P82-2008-2009NACO.2009-01-03T02:23:51.357_mean.fits")
#     with fits.open("/home/tdewacher/Documents/Stage/P82-2008-2009/Betelgeuse/0.007204/NB_1.26,/export/P82-2008-2009NACO.2009-01-03T02:23:51.357/P82-2008-2009NACO.2009-01-03T02:23:51.357_deconvolution.fits") as hdul:
#         ax[i].imshow(hdul[0].data,norm=PowerNorm(0.3),cmap="afmhot")
#         ax[i].set_title("sigma : " + str(sigma))
#         i +=1
        
# plt.show()