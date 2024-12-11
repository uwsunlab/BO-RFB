#%%
# Support for math
import numpy as np
import math

# Plotting tools
from matplotlib import pyplot as plt
import matplotlib

import warnings
warnings.filterwarnings('ignore')

#File Tools for local
import pandas as pd

import sys

import os
import fnmatch
import re

# %%
iteration_id = '102413'
directory = '/Users/ctuwsunlab/Documents/Sun_Lab/PNNL/PNNL_Iteration2/240805_102413 - Test 2B/'

# Get the CSV files only
files = fnmatch.filter(os.listdir(directory), '*.arw')
#files = fnmatch.filter(os.listdir('/Users/clarat/Documents/Sun_Lab/PNNL/102107 UV Spectra'), '*.csv')

# Sort files based on the second number in the filename
sorted_files = sorted(files, key=lambda s: int(re.findall(r'\d+', s)[1]) if len(re.findall(r'\d+', s)) >= 2 else 0)

# %%
def convert_arw_to_csv(filename,directory,i):
    with open(directory + "%s.arw"%(filename), 'r') as file:
        lines = file.readlines()[1:]

        data = [line.strip().split('	') for line in lines]

        # Convert to DataFrame
        df = pd.DataFrame(data[1:], columns=["X"+iteration_id+'_%s'%i, "Y"+ iteration_id+'_%s'%i])

        # Save to CSV
        df.to_csv(directory +'%s.csv'%(sorted_files[i].split(".")[0]),index=False)

#%%
#RUN ONLY ONCE
# Convert all .arw files in the directory 
for i in range(len(sorted_files)):
    filename = sorted_files[i].split(".")[0]
    convert_arw_to_csv(filename,directory,i)
# %%
