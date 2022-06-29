#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 23 15:59:40 2022

@author: tdewacher
"""

import numpy as np
import matplotlib.pyplot as plt
import os
import time
from astropy.io import fits
import pandas as pd
from matplotlib.colors import LogNorm, PowerNorm

R  = '\033[31m' # red
G  = '\033[32m' # green
P  = '\033[35m' # purple
W  = '\033[0m'  # white
B =  '\033[34m' # blue


def show_directory(path):
    n = len(os.listdir(path))
    for i in range(n):
        f = os.listdir(path)[i]
        if "fits" in f:
            with fits.open(path+"/"+f) as hdul:
                img = hdul[0].data
            plt.subplot(int(np.sqrt(n))+1,int(np.sqrt(n))+1,i+1)
            plt.imshow(img)
            print(f + " : " + str(np.mean(img)))
            plt.title(f)
    plt.show()
    
plt.close('all')
    
main = "/home/tdewacher/Documents/Stage/" 
# folders = ["P82-2008-2009","P88-2011-2012","P90-2012-2013","P94-2014-2015"]
folders = ["P94-2014-2015"]

start = time.time()
df = pd.DataFrame(data={"path":[],"epoch":[],"filter":[],"type":[]})

for f in folders:
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
        aldebaran = files[files["type"] == "Aldebaran"]
        
        m = max(len(betelgeuse),len(aldebaran))
    
        
        if m == 0: continue
    
        plt.ion()
        fig,ax = plt.subplots(2,m)
        
        for j in range(m):
            if j < len(betelgeuse): 
                with fits.open(betelgeuse["path"].to_numpy()[j]+"_mean.fits") as hdul:
                    ax[0,j].imshow(hdul[0].data,norm=PowerNorm(0.3),cmap="afmhot")
                    ax[0,j].set_title(betelgeuse["path"].to_numpy()[j].split('/')[-1][18:])
            if j < len(aldebaran): 
                with fits.open(aldebaran["path"].to_numpy()[j]+"_mean.fits") as hdul:
                    ax[1,j].imshow(hdul[0].data,norm=PowerNorm(0.3),cmap="afmhot")
                    ax[1,j].set_title(aldebaran["path"].to_numpy()[j].split('/')[-1][18:])               
        plt.suptitle(f + " : " + filt)
        plt.show(block=False)
        
        
        j = int(input("Betelgeuse id : "))
        
        with fits.open(betelgeuse["path"].to_numpy()[j]+"_mean.fits") as hdul:
            ax1[0,i].imshow(hdul[0].data,norm=PowerNorm(0.3),cmap="afmhot",interpolation="bicubic")
            ax1[0,i].set_title("Betelg - " + filt)
            
        j = int(input("Aldebaran id : "))
         
        with fits.open(aldebaran["path"].to_numpy()[j]+"_mean.fits") as hdul:
            ax1[1,i].imshow(hdul[0].data,norm=PowerNorm(0.3),cmap="afmhot",interpolation="bicubic")
            ax1[1,i].set_title("Aldeb - " + filt)
            
    
    plt.show()
             
        
    # max psf / max tache d'airy
    
    
    
    

