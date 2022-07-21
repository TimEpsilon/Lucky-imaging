#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 19 17:20:26 2022

@author: tdewacher
"""

from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
import astropy.units as u
import pandas as pd
from astroquery.vizier import Vizier
from scipy import interpolate
import astropy.units as u

#Inputs
home = Path.home()
infile = Path(home, 'Documents/Stage/Callibration/ckp00_4000.fits')
naco_dir = Path(home, 'Documents/Stage/Callibration/NACO_transmission/')
Johnson_file = Path(home, 'Documents/Stage/Callibration/Johnson_filters.txt')
theta_ald = 20.6 #mas

def getSED():
    '''
    Returns the SED interval and values

    Returns
    -------
    wlen : array
        interval.
    spec : array
        values.

    '''
    
    #Opening FITS file
    with fits.open(infile) as hdul:
        hdr = hdul[0].header
        sed_data = hdul[1].data
    
    #Getting SED
    wlen = sed_data['WAVELENGTH']
    spec = sed_data['g15']*(np.deg2rad(theta_ald*1e-3/(3600*2)))**2
    
    #Applying units
    wlen = wlen*u.AA
    spec = spec*u.erg/(u.s*u.cm**2*u.AA)
    
    #Unit conversion
    wlen = wlen.to('micron')
    spec = spec.to('W/(m2*micron)')
    
    return wlen,spec

def getNacoFilter(filt):
    
    #NACO filters
    naco_filt = naco_dir.glob('NB*txt')
    for cur_filt in naco_filt:
        wlen_filt, trans_filt = np.loadtxt(cur_filt, unpack=True)
        cur_filt = cur_filt.name
        cen_filt = float(cur_filt[cur_filt.find('NB')+3:cur_filt.find('.dat')])

        if filt in cur_filt:
            return wlen_filt, trans_filt
        
def getFluxTheorique(filt):
    '''
    Returns the flux intensity in theory for a given filter

    Parameters
    ----------
    filt : str
        name of filter.
    '''
    # Data
    spectrum,flux = getSED()
    band, absorp = getNacoFilter(filt)
    
    # Interpolate
    f = interpolate.interp1d(band,absorp,bounds_error=False,fill_value=0)
    inter_absorp = f(spectrum)
    
    # Spectral length
    length = (band[-1] - band[0])
    length *= u.micron
    result = np.trapz(inter_absorp*flux,x=spectrum) / length
    return result