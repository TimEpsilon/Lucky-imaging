#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 19 14:55:09 2022

A function to derive the flux of Aldebaran in the NACO filters
from the Castelli-Kurucz SEDs

T = 4000K
log g = 1.5
vt = 2km/s
solar mettalicity
https://www.stsci.edu/hst/instrumentation/reference-data-for-calibration-and-tools/astronomical-catalogs/castelli-and-kurucz-atlas

@author: mmontarges
"""

from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
import astropy.units as u
import pandas as pd
from astroquery.vizier import Vizier

plt.close('all')

#Inputs
home = Path.home()
infile = Path(home, 'Documents/Stage/Callibration/ckp00_3750.fits')
naco_dir = Path(home, 'Documents/Stage/Callibration/NACO_transmission/')
Johnson_file = Path(home, 'Documents/Stage/Callibration/Johnson_filters.txt')
datacube_82 = Path(home,"Documents/Stage/P82-2008-2009/Betel_AllFilters.fits")

theta_ald = 42 #mas

#Opening FITS file
with fits.open(infile) as hdul:
    hdr = hdul[0].header
    sed_data = hdul[1].data

#Getting SED
wlen = sed_data['WAVELENGTH']
spec = sed_data['g00']*(np.deg2rad(theta_ald*1e-3/(3600*2)))**2

#Applying units
wlen = wlen*u.AA
spec = spec*u.erg/(u.s*u.cm**2*u.AA)

#Unit conversion
wlen = wlen.to('micron')
spec = spec.to('W/(m2*micron)')

#Plot SED
fig, ax = plt.subplots(1, 1)
ax_filt = ax.twinx()
ax.plot(wlen, spec)

#NACO filters
naco_filt = naco_dir.glob('NB*txt')
max_lambda = []
for cur_filt in naco_filt:
    wlen_filt, trans_filt = np.loadtxt(cur_filt, unpack=True)
    cur_filt = cur_filt.name
    cen_filt = float(cur_filt[cur_filt.find('NB')+3:cur_filt.find('.dat')])

    #Plot NACO filters
    ax_filt.plot(wlen_filt, trans_filt, 'r',linewidth=0.5,linestyle="dashed")
    max_lambda.append(wlen_filt[trans_filt == np.max(trans_filt)][0])

#Ducati 2002 photometry
result = Vizier.query_object("Betelgeuse", catalog=["II/237/colors"])[0]
mag = np.array([result['B-V']+result['Vmag'], result['Vmag'], result['R-V']+result['Vmag'], result['I-V']+result['Vmag'], result['J-V']+result['Vmag'], result['H-V']+result['Vmag'], result['K-V']+result['Vmag']])
mag = np.array([cur_mag[0] for cur_mag in mag])

#Johnson filter values
df_Johnson = pd.read_csv(Johnson_file, comment='#')
wlen_ducati = np.zeros_like(mag)
flux_ducati = np.zeros_like(mag)
for n, cur_mag in enumerate(mag):
    flux_ducati[n] = df_Johnson['zero'][1+n]*10**(-cur_mag/2.5)
    wlen_ducati[n] = df_Johnson['lambda_0'][1+n]
flux_ducati = flux_ducati*u.erg/(u.s*u.cm**2*u.AA)
flux_ducati = flux_ducati.to('W/(m2*micron)')
ax.plot(wlen_ducati, flux_ducati, 'ok', ms=3)

#Cosmetics
ax.set_xscale('log')
ax.set_yscale('log')
ax.set_xlabel(r'Wavelength ($\mu$m)')
ax.set_ylabel(r'Spectral flux density (W m$^{-2}$ $\mu$m$^{-1}$)')
ax.grid()
ax.set_xlim(0.5, 3)
ax.set_ylim(1e-9, 1e-5)
ax_filt.set_ylabel('Filter transmission')
ax_filt.set_ylim(0, 1)


# Datacube P82
with fits.open(datacube_82) as hdul:
    img = hdul[0].data
t,dx,dy = np.shape(img)

# Radius map
X,Y = np.indices((dx,dy),dtype=float)
X -= dx/2.0
Y -= dy/2.0
R = np.hypot(X,Y)

# Flux
disc = []
for i in range(t):
    mask = np.logical_and(R<dx/2,R>dx/2-10)
    bckgr = np.median(img[i,:,:][mask])
    disc.append((img[i,:,:][R<=64] - bckgr).sum())
ax.plot(max_lambda,disc,"+",color="g")
print(disc)


