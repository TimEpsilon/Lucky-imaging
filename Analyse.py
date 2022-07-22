#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 12 15:55:42 2022

@author: tdewacher
"""

import numpy as np
import os
from astropy.io import fits
import matplotlib.pyplot as plt
from FinalImageSelection import get_deconvolution_datacube
import pandas as pd
from colour import Color
import SqrtScale
from matplotlib.colors import PowerNorm
import matplotlib.colors as mcolors
from image_registration.fft_tools import shift

plt.close('all')

def align(offset,image):
    """
    Aligns offset to image.

    Parameters
    ----------
    offset : 2D array
        The image to align.
    image : 2D array
        The reference image to align to.

    Returns
    -------
    corrected : 2D array
        The offset image after alignement with image
    """
    ymax, xmax = np.unravel_index(image.argmax(), image.shape)
    off_ymax, off_xmax = np.unravel_index(offset.argmax(), offset.shape)
    dx = xmax-off_xmax
    dy = ymax-off_ymax
    result = shift.shiftnd(offset,[dy,dx])
    return result
    

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
        image = np.zeros((np.shape(img)[0],dx,dy))
        for i in range(np.shape(img)[0]):
            image[i,:,:] = crop_image(img[i,:,:], dx, dy)
        return image
    
    else:
        Dy,Dx = np.shape(img)
        return img[int(Dy/2-dy/2):int(Dy/2+dy/2) , int(Dx/2-dx/2):int(Dx/2+dx/2)]

def getMeanBetweenRadius(img,r,R):
    dx,dy = np.shape(img)
    X,Y = np.indices((dx,dy),dtype=float)
    X -= dx/2 - 0.5
    Y -= dy/2 - 0.5
    radius = np.hypot(X,Y)
    
    # Calculates the mean 
    length = len(img[radius<=R]) - len(img[radius<r])
    if length == 0: 
        return 0
    return (img[radius<=R].sum() - img[radius<r].sum())/length

def getMeanBetweenAngle(img,theta1,theta2):
    '''
    Calculates the mean between theta2 > theta1.
    The angles are supposed to be between [0,2*pi]

    Parameters
    ----------
    img : 2D array
    theta1 : nbr
    theta2 : nbr

    Returns
    -------
    mean : nbr
    '''
    if theta2 > np.pi:
        theta2 -= 2*np.pi
        
    
    dx,dy = np.shape(img)
    X,Y = np.indices((dx,dy),dtype=float)
    X -= dx/2 - 0.5
    Y -= dy/2 - 0.5
    angle = np.arctan2(X,Y)
    
    # Calculates the mean 
    length = len(img[angle<=theta2]) - len(img[angle<=theta1])
    if length == 0: 
        return 0
    return (img[angle<=theta2].sum() - img[angle<=theta1].sum())/length


def getRadialProfile(path,n=30):
    # Open
    with fits.open(path) as hdul:
        img = hdul[0].data
        hdr = hdul[0].header
        
    # Radial map
    dx,dy = np.shape(img)
    R = np.hypot(dx,dy)/2
    r = np.linspace(0,R,n)
    dr = r[1]-r[0]
    
    # Sum
    somme = []
    for i in r:
        somme.append(getMeanBetweenRadius(img,i,i+dr))
        
    # to mas
    pxToAs = hdr["CD2_2"]*3600
    r *= pxToAs*1000
    filt = path.split('/')[-4]
    i = filters.index(filt)
    
    # Plot
    plt.figure(fig[i].number)
    plt.plot(r,np.array(somme)/(pxToAs**2),label=path.split(".")[-2][:3])
    plt.yscale("log")
    plt.vlines(43.7/2,np.min(somme),plt.gca().get_ylim()[1],color="red",linestyle="dashed")
    plt.xlabel("Radius (mas)")
    plt.ylabel("Intensite moyenne (W m-2 µm-1 as-2")
    plt.title(filt)
    plt.legend()
    plt.grid()
    
    
def PlotRadialDatacube(path,n=30):
    '''
    Plots the radial profile of every image in a datacube where each frame is a deconvolution iteration

    Parameters
    ----------
    path : str

    '''
    # Open image
    with fits.open(path) as hdul:
        img = hdul[0].data
        hdr = hdul[0].header
        
        # Radius
        t,dx,dy = np.shape(img)
        R = np.hypot(dx,dy)/2
        r = np.linspace(0,R,n)
        dr = r[1] - r[0]
        
        # Plotting color
        red = Color("red")
        colors = list(red.range_to(Color("black"),t))
        df = pd.read_csv("/home/tdewacher/Documents/Stage/P82-2008-2009/deconvolution_selection.csv")
        i = df[df["Path"].str.contains(path)]["ID"].to_numpy()[0]-1
        colors[i] = Color("green")
        
        
        # One plot per filter
        filt = path.split('/')[-4]
        index = filters.index(filt)
        
        figure = plt.figure(fig[10+index].number)
        
        # Dynamic subplot
        m = len(figure.axes)
        if m == 0:
            ax = figure.add_subplot(111)
        else : 
            for i in range(m):
                figure.axes[i].change_geometry(m+1,1,i+1)
                ax = figure.add_subplot(m+1,1,m+1)
                
        # For every frame of the datacube
        for j in range(t):
            somme = []
            for i in r:
                somme.append(getMeanBetweenRadius(img[j,:,:],i,i+dr))
            width = 0.5
            if colors[j] == Color("green"):
                width = 1
            ax.plot(r*hdr["CD2_2"]*1000*3600,somme,color=colors[j].get_web(),label=j,linewidth=width)
            ax.set_yscale("log")
            ax.set_title(path.split("/")[-1])
            ax.set_xlabel("Radius (mas)")
            ax.set_ylabel("Intensite relative")
            plt.suptitle(filt)
            plt.grid()
            plt.xlim(0,r[-1]*hdr["CD2_2"]*1000*3600)
            # plt.vlines(43.7,np.min(somme),plt.gca().get_ylim()[1],color="red",linestyle="dashed")
            plt.show()
            
def PlotDatacube(path,n=100):
    '''
    Plots the angular profile of every image in a datacube where each frame is a different wavelength

    Parameters
    ----------
    path : str

    '''
    # Open image
    with fits.open(path) as hdul:
        img = hdul[0].data
        hdr = hdul[0].header
        
        # Radius
        t,dx,dy = np.shape(img)
        R = np.hypot(dx,dy)/2
        r = np.linspace(0,R,n)
        dr = r[1] - r[0]
        
        
        red = Color("red")
        colors = list(red.range_to(Color("green"),t))
        if not hdr.get("CD2_2",default=None) is None:
            pxToMas = hdr["CD2_2"]
        else:
            pxToMas = 9.215275e-07
        
        # One plot per filter
        plt.figure()
                
        # For every frame of the datacube
        for j in range(t):
            somme = []
            for i in r:
                somme.append(getMeanBetweenRadius(img[j,:,:],i,i+dr))
            plt.plot(r*pxToMas*1000*3600,np.array(somme)/((pxToMas*3600)**2),label=filters[j])
        plt.vlines(43.7/2,np.min(somme),plt.gca().get_ylim()[1],color="red",linestyle="dashed")
        plt.yscale("sqrt")
        _,imax = plt.gca().get_ylim()
        di = imax - 0
        pxPercent = 95
        imax -= pxPercent/100 * di
        plt.ylim(0,imax)
        
        plt.title(path.split("/")[-1])
        plt.xlabel("Radius (mas)")
        plt.ylabel("Intensite moyenne (W m-2 µm-1 as-2)")
        plt.xlim(0,r[-1]*pxToMas*1000*3600)
        plt.legend()
        plt.show()

def substractEpoch(start,end):
    '''
    Substracts the start datacube from the end.
    Datacubes are supposed to be of same temporal size and already callibrated.

    Parameters
    ----------
    start : str
        path to datacube.
    end : str
        path to datacube.

    '''
    # Open images 
    with fits.open(start) as hdul:
        img_s = hdul[0].data
        hdr_s = hdul[0].header
        t1,dy1,dx1 = np.shape(img_s)
        pxToMas = hdr_s["CD2_2"]
        
    with fits.open(end) as hdul:
        img_e = hdul[0].data
        hdr_e = hdul[0].header
        t2,dy2,dx2 = np.shape(img_e) 
        
    # different temporal dimension
    if t1 != t2: 
        print("Not same size")
        return
    
    # Crop
    dx = min(dx1,dx2)
    dy = min(dy1,dy2)
    img_e = crop_image(img_e, dx, dy)
    img_s = crop_image(img_s, dx, dy)
    
    # Alignment and norm
    for i in range(t1):
        img_s[i,:,:] = align(img_s[i,:,:],img_e[i,:,:])
        img_s[i,:,:] /= np.max(img_s[i,:,:])
        img_e[i,:,:] /= np.max(img_e[i,:,:])
    
    # Substract
    img = img_e - img_s
    img[abs(img)<np.max(img)/10000] = 0
    
    for i in range(t1):
        fov = pxToMas*1000*3600/2*np.shape(img_e)[1]
        plt.figure()
        
        circle = plt.Circle((1.5,-1.5),43.7/2,color='green',linestyle="dashed",linewidth=0.5,fill=False)
        plt.subplot(131)
        length = np.max(img_e) - np.min(img_e)
        vmax = np.max(img_e) - 95/100 * length
        plt.imshow(img_e[i,:,:],norm=PowerNorm(0.5,vmax=vmax),cmap="afmhot",origin="lower",extent=[-fov,fov,-fov,fov],interpolation="bicubic")
        plt.suptitle(filters[i][:-1])
        plt.title("P86")
        plt.colorbar(label="sign( I ) * sqrt( |I| )")
        
        length = np.max(img_s) - np.min(img_s)
        vmax = np.max(img_s) - 95/100 * length
        plt.subplot(132)
        plt.imshow(img_s[i,:,:],norm=PowerNorm(0.5,vmax=vmax),cmap="afmhot",origin="lower",extent=[-fov,fov,-fov,fov],interpolation="bicubic")
        plt.title("P82")
        plt.colorbar(label="sign( I ) * sqrt( |I| )")
        
        plt.subplot(133)
        plt.imshow(sqrt_norm(img[i,:,:]),norm=mcolors.CenteredNorm(0),cmap="seismic",origin="lower",extent=[-fov,fov,-fov,fov])
        plt.title("P86 - P82")
        plt.colorbar(label="sign( I ) * sqrt( |I| )")
        plt.gcf().gca().add_patch(circle)
        plt.show()
        
        
    
    # Saving
    hdul = fits.PrimaryHDU(img)
    path = start[:-len(start.split('/')[-1])] + "difference_datacube.fits"
    hdul.writeto(path,overwrite=True)
    
    # Update hdr
    with fits.open(path,mode="update") as hdul:
        hdr = hdul[0].header
        hdr["CD2_2"] = pxToMas
        hdul.flush()
    

def sqrt_norm(a):
    result = np.sqrt(a)
    result[a<0] = - np.sqrt(abs(a[a<0]))
    return result
        
# =============================================================================
#                                   MAIN
# =============================================================================

    
directory = "/home/tdewacher/Documents/Stage/P82-2008-2009/"  
deconv = get_deconvolution_datacube(directory)

# One plot per filter
filters = ["NB_1.04,","NB_1.08,","NB_1.09,","NB_1.24,","NB_1.26,","NB_1.28,","NB_1.64,","NB_1.75,","NB_2.12,","NB_2.17,"]
colors = ["b","g","k","c","m","y","#8f3feb","#d92582","#a7b33e","#e89607"]
    
fig = [plt.figure(j) for j in range(20)]

for path in deconv:
    getRadialProfile(path.replace("deconvolution","final"),n=100)
    PlotRadialDatacube(path)

start = "/home/tdewacher/Documents/Stage/P82-2008-2009/Betel_AllFilters.fits"
end = "/home/tdewacher/Documents/Stage/P82-2008-2009/P86-Betel_AllFilters.fits"
substract = "/home/tdewacher/Documents/Stage/P82-2008-2009/difference_datacube.fits"
PlotDatacube(start)
PlotDatacube(end)

substractEpoch(start, end)

PlotDatacube(substract)

plt.show()