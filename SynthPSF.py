#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul  5 11:01:00 2022

@author: tdewacher
"""
import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
import pandas as pd
import os


def createSynthPSF(shape,pxToMas,name):
    '''
    Creates a synthetic psf for a given size

    Parameters
    ----------
    shape: tuple
    
    pxToMas : nbr
        1 px = pxToMas mas.
    '''    
    X,Y = np.indices(shape,dtype=float)
    dx,dy = shape
    X -= dx/2.0-0.5
    Y -= dy/2.0-0.5
    R = np.sqrt(X**2 + Y**2)
    omega = 20.6/2
    
    gamma =  R * pxToMas
    gamma[gamma>omega] = omega
    u = np.sqrt(1 - (np.sin(gamma/3600/1000*np.pi/180) / np.sin(omega/3600/1000*np.pi/180))**2)
    ext = extinction(u, coeff[name.split("-")[0]])
    ext[gamma>=omega] = 0
    
    hdul = fits.PrimaryHDU(ext)
    hdul.writeto(main+"/synthetic_psf/"+name,overwrite=True)
    
        
def extinction(u,a):
    '''
    Returns the extinction for u
    
    Parameters
    ----------
    u : nbr
        u = cos(gamma)
    a : array
        coeff of the extinction function
    '''
    
    somme = 1
    
    for i in range(4):
        somme -= a[i]*(1-np.power(u,(i+1)/2.0))
    
    return somme

def getBand(filtre):
    filtre = filtre[3:7]
    i = float(filtre)
    
    if i < 1.265+0.25/2: return "J"
    if i > 1.66-0.33/2 and i < 1.66+0.33/2: return "H"
    if i > 2.18-0.35/2: return "K"
    print("No band found for" + filtre +  "("+str(i)+"µm)")
    return
    

main = "/home/tdewacher/Documents/Stage/" 
folders = ["P82-2008-2009","P88-2011-2012","P90-2012-2013","P94-2014-2015"]
coeff = pd.DataFrame(data={"null":[0,0,0,0],"I":[1.2658,-1.9371,2.4284,-0.8895],"J":[0.7157,-0.5694,0.9465,-0.4094],"H":[0.8069,0.1187,-0.3620,0.1437],"K":[0.8900,-0.3487,0.0859,-0.0053]})


if not os.path.exists(main+"/synthetic_psf"):
    os.mkdir(main+"/synthetic_psf")

