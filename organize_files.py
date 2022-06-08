# -*- coding: utf-8 -*-
"""
Created on Tue May 31 16:41:59 2022

@author: timde
"""

from pathlib import Path
import matplotlib.pyplot as plt
from astropy.io import fits
import numpy as np
import os
import pandas as pd


def move_by_type(filename,location,destination,callib):
    """
    Moves every file in a given folder to the destination folder based on its type.
    
    Parameters
    ---------
    filename: str
        The path to the file obtained with the **save_file** method.
        
    location: str
        The path to the folder containing the dataset.
        
    destination: str
        The path to the folder where the callibration pictures will be moved.
    
    callib: str
        The object type.
    
    """
    
    # Create destination if not exist
    if not os.path.exists(destination):
        os.makedirs(destination)
        print(destination + " created !")
    
    # Dataframe creation
    f_list = pd.read_csv(filename,sep=';')
      
    # Dataframe of every row containing callib in lookingat
    callib_list = f_list.loc[f_list['LookingAt'].to_numpy() == callib]
    
    
    for f in callib_list["Name"].to_numpy():
        path = Path(location + "/" + f)
        if not path.is_file(): continue
        os.rename(location + "/" + f,destination + "/" + f)
        

def move_by_name(filename,location,destination,name):
    """
    Moves every file in a given folder to the destination folder based on its object name.
    
    Parameters
    ---------
    filename: str
        The path to the file obtained with the **save_file** method.
        
    location: str
        The path to the folder containing the dataset.
        
    destination: str
        The path to the folder where the callibration pictures will be moved.
        
    name: str
        The object name
    
    """
    
    # Create destination if not exist
    if not os.path.exists(destination):
        os.makedirs(destination)
        print(destination + " created !")
    
    # Dataframe creation
    f_list = pd.read_csv(filename,sep=';')
      
    # Dataframe of every row containing callib in lookingat
    callib_list = f_list.loc[f_list['Object'].to_numpy() == name]
    
    
    for f in callib_list["Name"].to_numpy():
        path = Path(location + "/" + f)
        if not path.is_file(): continue
        os.rename(location + "/" + f,destination + "/" + f)
        
   
def apply_filters_to_parent(filename,parent):
    """
    Sorts every child file by filters.

    Parameters
    ----------
    ilename: str
        The path to the file obtained with the **save_file** method.
        
    location: str
        The path to the folder containing the dataset.

    """
    for f in os.listdir(parent):
        if ".fits" in f:
            move_by_filter(filename, parent)
        else:
            child = os.path.join(parent, f)
            if os.path.isdir(child):
                apply_filters_to_parent(filename, child)

def move_by_filter(filename,location):
    """
    Moves every file in a given folder based on their filters.
    
    Parameters
    ---------
    filename: str
        The path to the file obtained with the **save_file** method.
        
    location: str
        The path to the folder containing the dataset.
        
    """
    # Dataframe creation
    f_list = pd.read_csv(filename,sep=';')
    
    # Unique filter list
    filters = np.unique(f_list["Filter"].to_numpy())
    
    for filt in filters:
        destination = os.path.join(location,filt)
        # Create destination if not exist
        if not os.path.exists(destination):
            os.makedirs(destination)
            print(destination + " created !")  
    
    for i,row in f_list.iterrows():
        f = row["Name"]
        destination = os.path.join(location,row["Filter"])
        path = Path(location + "/" + f)
        if not path.is_file(): continue
        os.rename(location + "/" + f,destination + "/" + f)
    
def subdivide_by_time(filename,location):
    """
    Creates and moves a group of files to folders based on their integration time

    Parameters
    ----------
    filename : str
        The path to the file obtained with the **save_file** method.
    location : str
        The path to the folder containing the dataset.

    """
    
    # Dataframe creation
    name = location.split("/")[-1]
    f_list = pd.read_csv(filename,sep=';')
    
    # Getting files in current folder
    f_list = f_list.loc[np.logical_or(f_list['Object'].to_numpy() == name,f_list['LookingAt'].to_numpy() == name)]
      
    # Dataframe of every row containing callib in lookingat
    time_list = np.unique(f_list['IntegrationTime'].to_numpy())
    
    # Iteration over every time
    for time in time_list:
        time = location+"/"+str(time)
        # Create destination if not exist
        if not os.path.exists(time):
            os.makedirs(time)
            print(time + " created !")
        
    # Move every file to corresponding time folder
    for i in f_list.index:
        f = f_list["Name"][i]
        t = str(f_list["IntegrationTime"][i])
        path = Path(location + "/" + f)
        if not path.is_file(): continue
        os.rename(location + "/" + f, location+ "/" + t + "/" + f)


main_folder = ["P82-2008-2009","P88-2011-2012","P90-2012-2013","P94-2014-2015"]
image_type = ["DARK","FLAT,SKY","FLAT,LAMP","SKY","STD"]
image_name = ["Betelgeuse","Aldebaran","31_Ori"]
master_type = ["DARK","SKY","FLAT,SKY"]
main_path = "C:/Users/timde/Documents/Stage"


# for folder in main_folder:
    
    # # Move and subdivide by name
    # for iname in image_name:
    #     move_by_name(folder+"_Output.csv", main_path+"/"+folder, main_path+"/"+folder+"/"+iname, iname)
    #     subdivide_by_time(folder+"_Output.csv", main_path+"/"+folder+"/"+iname)
        
        
    
    # # Move and subdivide by type
    # for itype in image_type:
    #     move_by_type(folder+"_Output.csv", main_path+"/"+folder, main_path+"/"+folder+"/"+itype, itype)
    #     subdivide_by_time(folder+"_Output.csv", main_path+"/"+folder+"/"+itype)
        
    # Move by filters
    # apply_filters_to_parent(main_path + "/" + folder+"_Output.csv", main_path+"/"+folder)
        
    
    # for master in master_type:
    #     # Create median file
    #     if not os.path.exists(main_path + "/" + folder + "/" + master): continue
    #     for time in os.listdir(main_path + "/" + folder + "/" + master):
    #         for exp_path in os.listdir(main_path + "/" + folder + "/" + master + "/" + time):
    #             save_median_to_file(main_path + "/" + folder + "/" + master + "/" + time + "/" + exp_path)
    #             print(main_path + "/" + folder + "/" + master + "/" + time + "/" + exp_path)
    #             master_dir = main_path + "/" + folder + "/" + master + "/" + time + "/" + exp_path + "/master"
    #             if not os.path.exists(master_dir): continue
            
    #             # Create master dark and sky
    #             if master != "FLAT,SKY":
    #                 save_master_dark_to_file(master_dir)
        
    #             else:
    #                 save_master_flat_to_file(master_dir)
                 
# save_clean_images("/home/Documents/Stage/P94-2014-2015/Betelgeuse/0.008717/NACO.2015-02-07T01_04_31.326.fits")