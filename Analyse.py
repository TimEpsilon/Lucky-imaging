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

plt.close('all')

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
    r *= hdr["CD2_2"]*3600*1000
    filt = path.split('/')[-4]
    i = filters.index(filt)
    
    # Plot
    plt.figure(fig[i].number)
    plt.plot(r,somme,label=path.split(".")[-2][:3])
    plt.yscale("log")
    plt.vlines(43.7,np.min(somme),plt.gca().get_ylim()[1],color="red",linestyle="dashed")
    plt.xlabel("Radius (mas)")
    plt.ylabel("Intensite moyenne (W m-2 µm-1 sr-1")
    plt.title(filt)
    plt.legend()
    
def getPolarProfile(path,n=30):
    # Open
    with fits.open(path) as hdul:
        img = hdul[0].data
        hdr = hdul[0].header
        
    # Polar map
    dx,dy = np.shape(img)
    theta_f = np.pi
    theta = np.linspace(-theta_f,theta_f,n)
    dtheta = theta[1]-theta[0]
    
    # Sum
    somme = []
    for i in theta:
        somme.append(getMeanBetweenAngle(img,i,i+dtheta))
    # Filter
    filt = path.split('/')[-4]
    i = filters.index(filt)
    
    # Plot
    plt.figure(fig[i].number)
    plt.subplot(122,projection="polar")
    plt.polar(theta,somme,label=path.split(".")[-2][:3])
    plt.ylabel("Intensite")
    plt.title(filt)
    plt.legend()
    
    
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
            ax.set_ylabel("Intensite")
            plt.suptitle(filt)
            # plt.vlines(43.7,np.min(somme),plt.gca().get_ylim()[1],color="red",linestyle="dashed")
            plt.show()
            
def PlotAngularDatacube(path,n=30):
    '''
    Plots the angular profile of every image in a datacube where each frame is a deconvolution iteration

    Parameters
    ----------
    path : str

    '''
    # Open image
    with fits.open(path) as hdul:
        img = hdul[0].data
        hdr = hdul[0].header
        
        # Angle
        t,dx,dy = np.shape(img)
        theta_f = np.pi
        theta = np.linspace(-theta_f,theta_f,n)
        dtheta = theta[1]-theta[0]
        
        # Plotting color
        red = Color("red")
        colors = list(red.range_to(Color("black"),t-1))
        colors.insert(0,Color("green"))
        
        # One plot per filter
        filt = path.split('/')[-4]
        index = filters.index(filt)
        
        figure = plt.figure(fig[20+index].number)
        
        # Dynamic subplot
        m = len(figure.axes)
        if m == 0:
            ax = figure.add_subplot(121,projection="polar")
        else : 
            for i in range(m):
                figure.axes[i].change_geometry(1,m+1,i+1)
                ax = figure.add_subplot(1,m+1,m+1,projection="polar")
                
        # For every frame of the datacube
        for j in range(t):
            somme = []
            for i in theta:
                somme.append(getMeanBetweenAngle(img[j,:,:],i,i+dtheta))
            if j == 0:
                ax.plot(theta,somme,color=colors[j].get_web(),label=j,linewidth=1)
            else:
                ax.plot(theta,somme,color=colors[j].get_web(),label=j,linewidth=0.5)
            ax.set_title(path.split("/")[-1])
            ax.set_ylabel("Intensite")
            plt.suptitle(filt)
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
            plt.plot(r*pxToMas*1000*3600,somme,label=filters[j])
        plt.vlines(43.7/2,np.min(somme),plt.gca().get_ylim()[1],color="red",linestyle="dashed")
        plt.yscale("sqrt")
        _,imax = plt.gca().get_ylim()
        di = imax - 0
        pxPercent = 95
        imax -= pxPercent/100 * di
        print(imax,di)
        plt.ylim(0,imax)
        
        plt.title(path.split("/")[-1])
        plt.xlabel("Radius (mas)")
        plt.ylabel("Intensite moyenne (W m-2 µm-1 sr-1)")
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
    
    # Substract
    img = img_e - img_s
    
    # Saving
    hdul = fits.PrimaryHDU(img)
    path = start[:-len(start.split('/')[-1])] + "difference_datacube.fits"
    hdul.writeto(path,overwrite=True)
    
    
        
# =============================================================================
#                                   MAIN
# =============================================================================

    
directory = "/home/tdewacher/Documents/Stage/P82-2008-2009/"  
deconv = get_deconvolution_datacube(directory)

# One plot per filter
filters = ["NB_1.04,","NB_1.08,","NB_1.09,","NB_1.24,","NB_1.26,","NB_1.28,","NB_1.64,","NB_1.75,","NB_2.12,","NB_2.17,"]
colors = ["b","g","k","c","m","y","#8f3feb","#d92582","#a7b33e","#e89607"]
    
# fig = [plt.figure(j) for j in range(20)]
fig = []

for path in deconv:
    # getRadialProfile(path.replace("deconvolution","final"),n=60)
    # getPolarProfile(path.replace("deconvolution","final"),n=60)
    # PlotRadialDatacube(path)
    # PlotAngularDatacube(path)
    a =1


start = "/home/tdewacher/Documents/Stage/P82-2008-2009/Betel_AllFilters.fits"
end = "/home/tdewacher/Documents/Stage/P82-2008-2009/P86-Betel_AllFilters.fits"
substract = "/home/tdewacher/Documents/Stage/P82-2008-2009/difference_datacube.fits"
PlotDatacube(start)
PlotDatacube(end)

substractEpoch(start, end)

PlotDatacube(substract)


plt.show()