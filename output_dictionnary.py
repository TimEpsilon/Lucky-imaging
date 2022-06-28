#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 27 16:16:33 2022

Start-up script to open/display FITS files

@author: mmontarges/timde
"""

from pathlib import Path
import matplotlib.pyplot as plt
from astropy.io import fits
import numpy as np
import pandas as pd
import os

def not_none(string):
    if string is None:
        return "None"
    else:
        return str(string)

def save_file(directory):
    """
    Saves the data (Name, Object, Looking at sky/object, Filter used, if a neutral density filter was used) 
    of every fit file of a given directory in a given file
    
    Parameters
    ------------
    directory: Path
        The path to the fits files
        
    filename: str
        The name of the output file. Used to write when called recursively.
    
    """
    with open(directory.name + "_Output.csv", "w") as text_file:
        # First Line
        text_file.write("Name;Object;LookingAt;Filter;IsNeutralDensity;IntegrationTime\n")
        
        for f in os.listdir(directory):
            if "master" in f or "median" in f or "clean" in f:
                continue
            input_file = os.path.join(directory, f)
                
            # Getting the data
            if ".fits" not in f: 
                df = get_dataframe_from_folder(input_file)
                
                for i,row in df.iterrows():
                    
                    text_file.write(row["Name"] + ";" + not_none(row["Object"]) + ";" + not_none(row["LookingAt"]) + ";" 
                                    + not_none(row["Filter"]) + ";" + not_none(row["IsNeutralDensity"]) + ";" 
                                    + not_none(row["IntegrationTime"]) + "\n")
                    
                continue
        
            with fits.open(input_file) as hdul1:
                img = np.array(hdul1[0].data)
                hdr = hdul1[0].header
                
                
                #getting the filter and if a nd filter was used
                filters = ""
                if hdr.get("HIERARCH ESO INS OPTI4 TYPE") == "FILTER": filters += hdr.get("HIERARCH ESO INS OPTI4 NAME") + ','
                if hdr.get("HIERARCH ESO INS OPTI5 TYPE") == "FILTER": filters += hdr.get("HIERARCH ESO INS OPTI5 NAME") + ','
                if hdr.get("HIERARCH ESO INS OPTI6 TYPE") == "FILTER": filters += hdr.get("HIERARCH ESO INS OPTI6 NAME") + ','
                filters += ";" + str(hdr.get("HIERARCH ESO INS OPTI3 NAME") == "ND_Long")
                
                text_file.write(str(f) + ";" 
                                + hdr.get('OBJECT') + ";" 
                                + hdr.get("HIERARCH ESO DPR TYPE") +";" 
                                + filters + ";" 
                                + str(hdr.get("EXPTIME")) + "\n")


def get_dataframe_from_folder(directory):
    """
    Returns a pandas dataframe from a given folder.

    Parameters
    ----------
    directory : str
        Path to the folder.

    Returns
    -------
    df: dataframe
        Contains the output of every child file.

    """
    
    df = pd.DataFrame({"Name":[],"Object":[],"LookingAt":[],"Filter":[],"IsNeutralDensity":[],"IntegrationTime":[]})
    
    for f in os.listdir(directory):
        if "master" in f or "median" in f or "clean" in f:
            continue
        input_file = os.path.join(directory, f)
        print(input_file)
        
        if ".fits" not in f:
            df_b = get_dataframe_from_folder(input_file)
            df = pd.concat([df,df_b])
        else:
            with fits.open(input_file) as hdul:
                hdr = hdul[0].header
                
                filters = ""
                if hdr.get("HIERARCH ESO INS OPTI4 TYPE") == "FILTER": filters += hdr.get("HIERARCH ESO INS OPTI4 NAME") + ','
                if hdr.get("HIERARCH ESO INS OPTI5 TYPE") == "FILTER": filters += hdr.get("HIERARCH ESO INS OPTI5 NAME") + ','
                if hdr.get("HIERARCH ESO INS OPTI6 TYPE") == "FILTER": filters += hdr.get("HIERARCH ESO INS OPTI6 NAME") + ','
                
                
                df_b = pd.DataFrame({"Name":[f],
                                    "Object":[hdr.get("OBJECT")],
                                    "LookingAt":[hdr.get("HIERARCH ESO DPR TYPE")],
                                    "Filter":[filters],
                                    "IsNeutralDensity":[str(hdr.get("HIERARCH ESO INS OPTI3 NAME") == "ND_Long")],
                                    "IntegrationTime":[str(hdr.get("EXPTIME"))]})
                df = pd.concat([df,df_b])
                                    
    return df


def remove_empty_folders(directory):
    """
    Removes any empty folder inside directory.

    Parameters
    ----------
    directory : str
        Path to the folder.


    """
    
    if ".fits" in directory:
        return
    
    if len(os.listdir(directory)) == 0:
        os.removedirs(directory)
        return
    
    for f in os.listdir(directory):
        if not ".fits" in f:
            child = os.path.join(directory,f)
            remove_empty_folders(child)
            
            
def move_to_main_folder(directory,main):
    """
    Moves every fits file to the main folder.

    Parameters
    ----------
    directory : str
        Path to the folder.
        
    main : str
        Folder where to move the files.


    """
    
    if ".fits" in directory or ".csv" in directory:
        os.rename(directory, main + "/" + directory.split('/')[-1])
        return
    
    for f in os.listdir(directory):
        child = os.path.join(directory,f)
        move_to_main_folder(child, main)
        
            



home = Path.home() 
print(home)


# directory = Path(home, "Documents/Stage/P94-2014-2015")
# save_file(directory)

# directory = Path(home, "Documents/Stage/P82-2008-2009")
# save_file(directory)
    
# directory = Path(home, "Documents/Stage/P88-2011-2012")
# save_file(directory)

# directory = Path(home, "Documents/Stage/P90-2012-2013")
# save_file(directory)

directory = Path(home, "Documents/Stage/2009_to_sort")
save_file(directory)

main_folder = ["P82-2008-2009","P88-2011-2012","P90-2012-2013","P94-2014-2015"]
main_path = "/home/tdewacher/Documents/Stage/"

# for folder in main_folder:
#     move_to_main_folder(main_path + folder, main_path + folder)
#     remove_empty_folders(main_path + folder)

#Example for nice plots
# fov = hdr['CD2_2']*hdr['NAXIS2']/2
# fov *= 3600*1000

# Plot
#fig, ax = plt.subplots(1, 1)
#pl1 = ax.imshow(img[1,:,:], origin='lower', cmap='afmhot', extent=[fov, -fov, -fov, fov])

