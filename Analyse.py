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

plt.close('all')

def getMeanBetween(img,r,R):
    dx,dy = np.shape(img)
    X,Y = np.indices((dx,dy),dtype=float)
    X -= dx/2 - 0.5
    Y -= dy/2 - 0.5
    radius = np.hypot(X,Y)
    
    length = len(img[radius<=R]) - len(img[radius<r])
    if length == 0: 
        return 0
    return (img[radius<=R].sum() - img[radius<r].sum())/length

def getRadioProfile(path,n=30):
    
    with fits.open(path) as hdul:
        img = hdul[0].data
        hdr = hdul[0].header
        
    dx,dy = np.shape(img)
    R = np.hypot(dx,dy)
    r = np.linspace(0,R,n)
    dr = R/n
    
    somme = []
    for i in r:
        somme.append(getMeanBetween(img,i,i+dr))
        
    r *= hdr["CD2_2"]*3600*1000
    color = "b"
    colors = ["b","g","k","c","m","y","#8f3feb","#d92582","#a7b33e","#e89607"]
    filters = ["NB_1.04,","NB_1.08,","NB_1.09,","NB_1.24,","NB_1.26,","NB_1.28,","NB_1.64,","NB_1.75,","NB_2.12,","NB_2.17,"]
    filt = path.split('/')[-4]
    
    for i,f in enumerate(filters):
        if filt in f:
            color = colors[i]
            break
        
    
    
    
    # plt.figure()
    plt.plot(r,somme/np.max(somme))
    plt.yscale("log")
    plt.vlines(45,0,1,color=color,label=filt)
    plt.xlabel("Radius (mas)")
    plt.ylabel("Intensite")
    # plt.show()
    
    
directory = "/home/tdewacher/Documents/Stage/P82-2008-2009/"  
deconv = get_deconvolution_datacube(directory)

for path in deconv:
    getRadioProfile(path.replace("deconvolution","final"),n=30)
    
plt.legend()
plt.show()