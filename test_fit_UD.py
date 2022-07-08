#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul  8 13:14:53 2022

@author: mmontarges
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

# from test_fwhm import chi2

def UD_map(XY, theta, A):

    radius = np.hypot(XY[0], XY[1])
    img = np.zeros_like(radius)
    img[radius<theta/2] = A

    return img

def UD_1D(XY, theta, A):

    return np.ravel(UD_map(XY, theta, A))

def LDD_map(XY, LDD_diam,  LDD_power):
    """
    This function returns a map of a LD power-law disk I = mu*alpha

    Parameters
    ----------
    **XY**

    **LDD_diam** : Stellar angular diameter in mas

    **LDD_power** : power law exponent
    """

    radius = np.hypot(XY[0], XY[1])
    intensity = np.zeros_like(radius)
    star_disk = (radius <= LDD_diam/2.)
    beta = np.arcsin(radius[star_disk]/(LDD_diam/2.))
    intensity[star_disk] = (np.cos(beta))**LDD_power

    return intensity

def LDD_1D(XY, LDD_diam, LDD_power):

    return np.ravel(LDD_map(XY, LDD_diam, LDD_power))


# =============================================================================
#                                    MAIN
# =============================================================================

"""plt.close('all')

#Inputs
npix = 100
pix_size = 3e-7 #deg
diam = 42
alpha = 0

#Generation of SCI image
Y, X = np.indices([npix, npix])
X = (X-npix/2)*pix_size*3600*1e3
Y = (Y-npix/2)*pix_size*3600*1e3
img = LDD_map((X, Y), diam, alpha)

#Fit
init_param = [80, 0.9]
opt, cov = curve_fit(LDD_1D, (X, Y), np.ravel(img), p0=init_param)
print(opt[0], cov[0, 0])
print(opt[1], cov[1, 1])

# Ini/Fitted map
img_init = LDD_map([X, Y], init_param[0], init_param[1])
img_fit = LDD_map([X, Y], opt[0], opt[1])

#Plot
fov = X.max()/2
fig, axes = plt.subplots(1, 3)
pl1 = axes[0].imshow(img, origin='lower', cmap='magma', extent=[fov, -fov, -fov, fov])
fig.colorbar(pl1, ax=axes[0])
axes[0].set_title('Oiginal image')
pl2 = axes[1].imshow(img_init, origin='lower', cmap='magma', extent=[fov, -fov, -fov, fov])
fig.colorbar(pl2, ax=axes[1])
axes[1].set_title('Init image')
pl3 = axes[2].imshow(img_fit, origin='lower', cmap='magma', extent=[fov, -fov, -fov, fov])
fig.colorbar(pl3, ax=axes[2])
axes[2].set_title('Fitted image')


chi = []
R = np.linspace(0,100,200)
for r in R:
    img_fit = LDD_map([X,Y], r, 0)
    chi.append(chi2(img_init,img_fit))
    
plt.figure()
plt.plot(R,chi)
plt.show()d"""
