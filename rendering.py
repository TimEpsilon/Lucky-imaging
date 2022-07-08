#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul  4 14:15:42 2022

@author: tdewacher
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

path = "/home/tdewacher/Documents/Stage/"
path += "P94-2014-2015/Betelgeuse/0.008717/NB_3.74,/export/P94-2014-2015NACO.2015-02-07T01:04:31.326/"
path += "P94-2014-2015NACO.2015-02-07T01:04:31.326_fwhm.csv"

df = pd.read_csv(path)

plt.figure()
plt.hist(df["0"],bins=100)
# plt.xlim(0,5)
plt.show()