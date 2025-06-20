# %% [markdown]
'''
# PNNL Spectra Extraction for Round 1
'''
# %% [code]
# Support for math
import numpy as np

# Plotting tools
from matplotlib import pyplot as plt
import matplotlib as mplt

import warnings
warnings.filterwarnings('ignore')

#Data frames tools
import pandas as pd
import os
import sys

from scipy.signal import find_peaks
from hplc.quant import Chromatogram

import fnmatch
import re

# %%
df_summary = pd.read_excel("208590_ESMI Synthesis EXPERIMENT.xlsx",sheet_name='2006')
df_summary.head()
# %%
file_directory = '102107 UV Spectra'
# Get the CSV files only
files = fnmatch.filter(os.listdir(file_directory), '*.csv')
#files = fnmatch.filter(os.listdir('/Users/clarat/Documents/Sun_Lab/PNNL/102107 UV Spectra'), '*.csv')

# Sort files based on the second number in the filename
sorted_files = sorted(files, key=lambda s: int(re.findall(r'\d+', s)[1]) if len(re.findall(r'\d+', s)) >= 2 else 0)
print(sorted_files)

#%% 
fig, axs = plt.subplots(1, 3, figsize=(12, 4))
n = 15
for idx, i in enumerate(range(3*n, 3*(n+1))):
    test1 = pd.read_csv(file_directory+'/'+sorted_files[i], skiprows=1)[['ACQUITY TUV ChA', 'ACQUITY TUV ChA 280nm']]
    test1.columns = ['wave', 'intensity']
    axs[idx].plot(test1['wave'], test1['intensity'], 'tab:red')
    axs[idx].set_xlabel('Time (min)')
    axs[idx].set_ylabel('Intensity')
    axs[idx].set_title(f'{sorted_files[i]}')
plt.tight_layout()
# %%
test1 = pd.read_csv(file_directory+'/'+sorted_files[3*n], skiprows=1)[['ACQUITY TUV ChA', 'ACQUITY TUV ChA 280nm']]
test2 = pd.read_csv(file_directory+'/'+sorted_files[3*n+1], skiprows=1)[['ACQUITY TUV ChA', 'ACQUITY TUV ChA 280nm']]
test3 = pd.read_csv(file_directory+'/'+sorted_files[3*n+2], skiprows=1)[['ACQUITY TUV ChA', 'ACQUITY TUV ChA 280nm']]
test3.columns = ['wave3', 'intensity3']
test1.columns = ['wave1', 'intensity1']   
test2.columns = ['wave2', 'intensity2']
test_df = pd.concat([test1, test2, test3], axis= 1)
test_df

# %%
# chrom = Chromatogram(test1, cols={'time':f'wave{1}', 'signal':f'intensity{1}'})
# chrom.fit_peaks(prominence=0.05)
peak_list = []
for i in range(1,3+1):
    chrom = Chromatogram(test_df, cols={'time':f'wave{i}', 'signal':f'intensity{i}'})
    chrom.correct_baseline()
    peak_list.append(chrom.fit_peaks(prominence=0.0005))
peak_list
# %%
def select_peaks_area(value,peak_list):
    array = np.zeros(len(peak_list))
    for i in range(len(peak_list)):
        if (peak_list[i]['retention_time'] == value).any() == True:
            indices = np.where(peak_list[i] == value)
            array[i] = peak_list[i]['area'].iloc[indices[0].item()]
        else:
            array[i] = 0
    return array

product_area = select_peaks_area(0.3,peak_list)
reactant_area = select_peaks_area(3.9,peak_list )+select_peaks_area(3.8,peak_list)


product_area , reactant_area
#%% 
total = product_area + reactant_area
print(product_area, total)

yield_prod = product_area/ total
yield_react = reactant_area/ total
print(yield_prod)

# %%
mean = np.mean(yield_prod)
std = np.std(yield_prod)
# %%
print(f"Mean Yield: {mean:1f}, Standard Deviation: {std:1f}")

# %%
