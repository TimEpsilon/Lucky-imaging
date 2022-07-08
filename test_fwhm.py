#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul  6 16:52:44 2022

@author: tdewacher
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from astropy.io import fits


FWHM_TO_SIGMA = 2*np.sqrt(2*np.log(2))

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
    """
    Gets the fwhm from a gaussian fit
    We suppose that 1px = 1mas, the image needs to be resized accordingly beforehand
    
    Parameters
    ----------
    img : 2D array

    Returns
    -------
    fwhm: nbr
        the fwhm in pixels.

    """
    
    guess = [np.max(img),40,0,0,0]
    
    y,x = np.indices(np.shape(img))
    x = x - np.shape(x)[1]/2
    x = x/np.max(x)
    y = y - np.shape(y)[0]/2
    
    try:
        popt, pcov = curve_fit(Gauss2D_ravel,(x,y),img.ravel(),p0=guess)
    except RuntimeError:
        popt = [0,0,0,0,0]
        pass
    
    return abs(popt[1])

def GaussMap(x,y,fwhm):
    s = fwhm/FWHM_TO_SIGMA
    return np.exp(-(x**2 + y**2)/(2*s**2))

def createSynthPSF(X,Y,name):
    '''
    Creates a synthetic psf for a given size
    We suppose 1px = 1mas
    Parameters
    ----------
    shape: tuple
    
    pxToMas : nbr
        1 px = pxToMas mas.
    '''    
    R = np.sqrt(X**2 + Y**2)
    omega = 47/2
    
    gamma = R
    gamma[gamma>omega] = omega
    u = np.sqrt(1 - (np.sin(gamma/3600/1000*np.pi/180) / np.sin(omega/3600/1000*np.pi/180))**2)
    ext = dark_dimming(u, [-0.0835,2.6921,-3.3510,1.3517])
    ext[gamma>=omega] = 0
    return ext
    
        
def dark_dimming(u,a):
    '''
    Returns the dark_dimming for u
    
    Parameters
    ----------
    u : nbr
        u = cos(gamma)
    a : array
        coeff of the dark_dimming function
    '''
    
    somme = 1
    
    for i in range(4):
        somme -= a[i]*(1-np.power(u,(i+1)/2.0))
    
    return somme

 
def UniformStar(XY,r):
    """
    Fitting model for an uniform disc

    Parameters
    ----------
    XY : [x,y]
    r : radius.

    Returns
    -------
    disc

    """
    x,y = XY
    R = np.sqrt(x**2 + y**2)
    unif = np.zeros_like(R)
    unif[R<r] = 1
    return unif

def UniformStar_ravel(XY,r):
    return np.ravel(UniformStar(XY, r))

def UniformStarMap(X,Y,r):
    R = np.sqrt(X**2 + Y**2)
    unif = np.zeros_like(R)
    unif[R<r/2] = 1
    return unif 

def getRadius_UniformStar(img):
    """
    Returns the radius from fitting a uniform disk
    
    Parameters
    ----------
    img : 2D array
        
    Returns
    -------
    popt : The parameters after fitting

    """
    guess = [50]
    
    y,x = np.indices(np.shape(img),dtype=float)
    x = x - np.shape(x)[1]/2+0.5
    y = y - np.shape(y)[0]/2+0.5
    
    
    popt, pcov = curve_fit(UniformStar_ravel,(x,y),img.ravel(),p0=guess)
    
    return popt

def chi2(model,img):
    '''
    Calculates chi² between the two images

    Parameters
    ----------
    model : 2D array
    img : 2D array of same size

    Returns
    -------
    chi2 : nbr

    '''    
    chi2 = np.sum((img-model)**2)/model.size
        
    return chi2


plt.close('all')

with fits.open("/home/tdewacher/Documents/Stage/P82-2008-2009/Betelgeuse/0.007204/NB_1.24,/export/P82-2008-2009NACO.2009-01-03T02:26:46.654/P82-2008-2009NACO.2009-01-03T02:26:46.654_norm.fits") as hdul:
    hdr = hdul[0].header
    img = hdul[0].data

# Fake 10x10 "real" pixel images, scaled up to be 1px = 1 mas
fov = 10*hdr["CD2_2"]*1000*3600/2



fig = plt.figure()

X,Y = np.indices((round(2*fov),round(2*fov)),dtype=float)
X -= np.shape(X)[1]/2-0.5
Y -= np.shape(Y)[0]/2-0.5

# Fake Star Limb dark
# img = createSynthPSF(X, Y, "")
# img += np.random.normal(size=np.shape(img),scale=0.1)

# Fake Star Uniform
img = np.zeros_like(X)
img[np.sqrt(X**2+Y**2)<47] = 1

# Plot fake star
R = np.sqrt(X**2+Y**2)
plt.subplot(2,3,1)
plt.imshow(img,extent=[fov,-fov,-fov,fov])
plt.title("Fake Star")
plt.subplot(2,3,(4,6))
plt.plot(np.linspace(-fov,fov,len(img[0,:])),img[round(fov),:],label="Fake Star",linewidth=2)
plt.title("Slice at middle of image")
plt.xlabel("mas")
plt.hlines(0.5, -fov, fov,color='red')

# Gaussian fit
fwhm = getFwhm(img)
print("FWHM from gaussian : " + str(fwhm))
fit = GaussMap(X, Y, fwhm)
plt.subplot(2,3,2)
plt.imshow(fit,extent=[fov,-fov,-fov,fov])
plt.title("Fitted Gaussian")
plt.subplot(2,3,(4,6))
plt.plot(np.linspace(-fov,fov,len(img[0,:])),fit[round(fov),:],label="Fitted Gaussian",linestyle="dashed",linewidth="1")

# Uniform fit
plt.subplot(2,3,3)
coeff = getRadius_UniformStar(img)
print("Radius from disk : " + str(coeff))
uniform_fit = UniformStarMap(X, Y, coeff[0])
plt.imshow(uniform_fit,extent=[fov,-fov,-fov,fov],vmin=0)
plt.title("Fitted Uniform")
plt.subplot(2,3,(4,6))
plt.plot(np.linspace(-fov,fov,len(img[0,:])),uniform_fit[round(fov),:],label="Fitted Uniform",linestyle="dashed",linewidth="1")

# plt.xlim(-50,50)
plt.legend()


plt.show(block=False)

# Chi2 plot
chi = []
R = np.linspace(0,100,200)
for r in R:
    disc = UniformStarMap(X, Y, 2*r)
    chi.append(chi2(img,disc))
    
plt.figure()
plt.plot(R,chi)
plt.show(block=False)

