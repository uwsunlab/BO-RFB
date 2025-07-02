# %% [markdown]
'''
# PNNL Spectra Extraction for Round 1
This notebook extracts the spectra from the HPLC data files and processes them to obtain the relevant chromatographic information.
It reads the HPLC data files, processes them to extract peaks, and saves the results in a CSV file.
The data is then used to analyze the yield of the product and reactant based
'''
# %% [code]
'''
Import the libraires
'''
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
'''
Load the summary data from the excel file
'''
df_summary = pd.read_excel("208590_ESMI Synthesis EXPERIMENT.xlsx",sheet_name='2006')
df_summary.head()
# %%
'''
file_directory: Directory where the HPLC data files are stored. (the string of the file name)
'''
file_directory = '102119 HPLC Data'
# Get the CSV files only
files = fnmatch.filter(os.listdir(file_directory), '*.csv')
#files = fnmatch.filter(os.listdir('/Users/clarat/Documents/Sun_Lab/PNNL/102107 UV Spectra'), '*.csv')

# Sort files based on the second number in the filename
sorted_files = sorted(files, key=lambda s: int(re.findall(r'\d+', s)[1]) if len(re.findall(r'\d+', s)) >= 2 else 0)
print(sorted_files)

# %%
#read them into pandas
columns_list = [pd.read_csv(file_directory +'/' + files,skiprows=1, sep='\t').columns.item() for files in sorted_files]

df_list = [pd.read_csv(file_directory+'/'+ sorted_files[i],skiprows=2, sep='\t',names=['X%s'%columns_list[i],'Y%s'%columns_list[i]]).dropna(axis=1, how='all') for i in range(len(sorted_files))]
df_list

# %%
'''
Put all the dataframes together into one big dataframe
This will create a dataframe with columns named X1, Y1, X2, Y2
'''
big_df = pd.concat(df_list,axis=1)
big_df
# %%
def plot_spec(id):
  data = [big_df['X%s'%columns_list[id]],big_df['Y%s'%columns_list[id]]]
  peaks, _ = find_peaks(data[1], prominence=0.005)
  col = (np.random.random(), np.random.random(), np.random.random())
  plt.plot(data[0],data[1])

  plt.vlines(data[0].values[peaks], 0, np.max(data[1]), linestyle='--', color='tab:grey')#'dodgerblue'

  plt.title('%s'%columns_list[id])#len(peaks))

plot_spec(0)
plt.xlabel('Wavelength (nm)')
plt.ylabel('Absorbance (mAU)')
# %%
'''
Plot the results of the spectra
'''
#plot the normalized spectra
fig = plt.figure(figsize=(20,25))
#Spectra vs Weights
for i in range(0,len(df_list)):
  plt.subplot(1+len(df_list)//6,6,i+1)
  plot_spec(i)
# %%
# %%
'''
Fit the peaks in the chromatogram
'''
def fit_peak_range(min, max, filter):
    '''
    min: minimum time to consider for peak detection 
    max: maximum time to consider for peak detection
    filter: minimum prominence of the peaks to be detected
    '''
    peak_list = []
    for i in range(big_df.shape[1]//2):
        chrom = Chromatogram(big_df, cols={'time':'X%s'%columns_list[i], 'signal':'Y%s'%columns_list[i]},time_window=[min,max])
        chrom.correct_baseline()
        peak_list.append(chrom.fit_peaks(prominence=filter))

    return peak_list
# %%
# Get the list of all peaks in the chromatogram
list_all = fit_peak_range(0, 7, 0.02)
list_all
# %%
# get the list of peaks in the product
list_product = fit_peak_range(0, 0.5, 0.05)
# %%
# ge
list_reactant = fit_peak_range(3.5, 4, 0.2)


#%% [markdown]
'''
>## NOTE
> - Product = 0.29 min RT
> - Reactant = 3.92 min RT
> - Sulfuric Acid = 5.25 min RT 
\
\
> All other peaks are potential side products \
> Area counts for compounds can assume intersection at origin and 8910360 = 0.053 g/ml

'''

# %%
def select_peaks_area(value, peak_list):
    '''
    value: retention time to select the peaks
    peak_list: list of peaks to select from
    '''
    array = np.zeros(len(peak_list))
    for i in range(len(peak_list)):
        if (peak_list[i]['retention_time'] == value).any() == True:
            indices = np.where(peak_list[i] == value)
            array[i] = peak_list[i]['area'].iloc[indices[0].item()]
        else:
            array[i] = 0
    return array

product_area = select_peaks_area(0.2,list_product)
reactant_area = select_peaks_area(3.9,list_reactant )+select_peaks_area(3.8,list_reactant )


# %%
time = np.array([element for element in  df_summary['Sample Time (min)']])
time = np.hstack([time[:21],time[24:]])

temp = np.array([element for element in  df_summary['Temperature (degC)']])
temp = np.hstack([temp[:21],temp[24:]])

sulfonating_agent= np.array([element for element in  df_summary['Sulfonating Agent\n(wt%)']])
sulfonating_agent= np.hstack([sulfonating_agent[:21],sulfonating_agent[24:]])

analyte = np.array([element for element in  df_summary['Reagent Ratio\n(mg/mL reagent/sulfonating agent)']])
analyte= np.hstack([analyte[:21],analyte[24:]])

product = np.hstack([product_area[:21],product_area[24:]])
reactant = np.hstack([reactant_area[:21],reactant_area[24:]])

# calculate the yield of the product and reactant
total = product + reactant 
yield_prod = product/ total
yield_react = reactant/ total


# %%
# Save the data into a dataframe
data_102119 = pd.DataFrame({
    '01_time': time,
    '01_temp': temp,
    '01_sulf': sulfonating_agent,
    '01_anly': analyte,
    '01_yield product': yield_prod,
})
data_102119.head(11)

# %%
# Export the data to a csv file
df_save = data_102119
list(df_save.columns)
df_save.to_csv('extracted_data_round1.csv', index=False)

# %% [markdown]
'''
> ### Data for 102119 HPLC Data
'''

# %%
