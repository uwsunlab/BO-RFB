#%%__markdown
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

from scipy.signal import find_peaks

from hplc.quant import Chromatogram
# %%
directory = '102728 HPLC Data.xlsx'
df_summary = pd.read_excel(directory,sheet_name='all conditions')
df_summary.head(7)

big_df = pd.read_excel(directory,sheet_name='HPLC Summary')
big_df 
#%%
big_df.columns[0]
# %%
# Plot the figure
fig, axs = plt.subplots(nrows=4, ncols=6, figsize=(15, 10))

for i in range(0, big_df.shape[1], 2):
    row = i // 12
    col = (i // 2) % 6
    axs[row, col].plot(big_df.iloc[:, i], big_df.iloc[:, i+1])
    #axs[row, col].set_title(f'{big_df.columns[i]} vs {big_df.columns[i+1]}')
    axs[row, col].set_xlabel(big_df.columns[i])
    axs[row, col].set_ylabel(big_df.columns[i+1])
    axs[row, col].grid(True)
plt.tight_layout()
plt.show()

# %%
def fit_peak_range(df, min, max,filter):
    peak_list = []
    for i in range(0,df.shape[1], 2):
        chrom = Chromatogram(df, cols={'time':big_df.columns[i], 'signal':big_df.columns[i+1]},time_window=[min,max])
        
        chrom.correct_baseline()

        peak_list.append(chrom.fit_peaks(prominence=filter))

    return peak_list
# %%
list_product = fit_peak_range(big_df,1.0, 2.0, 0.91)
list_product
# %%

list_reactant = fit_peak_range(big_df,2.1, 3.0, 0.2)
list_reactant 

#%% [markdown]
'''
# NOTE
- Product = 0.29 min RT (between 0:1)
- Reactant = 3.92 min RT  (between 0:1)
- Sulfuric Acid = 5.25 min RT  (between 3.5:4)
\
\
All other peaks are potential side products \
Area counts for compounds can assume intersection at origin and 8910360 = 0.053 g/ml

'''
def minmax(list_,name):
    min = list_[0][name].min()
    max = list_[0][name].max()
    for i in range(len(list_)):
        if min > list_[i][name].min():
            min =  list_[i][name].min()
        if max <  list_[i][name].max():
            max =  list_[i][name].max()

    return print(min,max)

minmax(list_product, 'retention_time')
minmax(list_reactant, 'retention_time')
# minmax(list_acid, 'retention_time')

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

def area_select(peak_list):
    array = np.zeros(len(peak_list))
    for i in range(len(peak_list)): 
        for j in range(len(peak_list[i])):
            iddf = peak_list[i]['retention_time'].argmin()
            val = peak_list[i]['area'].iloc[iddf].item()
        array[i] = val
    return array

def sum_area(peak_list):
    array = np.zeros(len(peak_list))
    for i in range(len(peak_list)): 
        for j in range(len(peak_list[i])):
            val = peak_list[i]['area'].sum()
        array[i] = val
    return array

#%%
reactant_region_area = area_select(list_reactant)
product_region_area = sum_area(list_product)
reactant_region_area ,product_region_area 
# %%
time = np.array([element for element in  df_summary['time']])
time = time[:] #np.hstack([time[:21],time[24:]])

temp = np.array([element for element in  df_summary['temp']])
temp = temp[:] #np.hstack([temp[:21],temp[24:]])

sulfonating_agent= np.array([element for element in  df_summary['Sulfonating Agent']])
sulfonating_agent= sulfonating_agent[:] #np.hstack([sulfonating_agent[:21],sulfonating_agent[24:]])

analyte = np.array([element for element in  df_summary['Analyte']])
analyte= analyte[:] #np.hstack([analyte[:21],analyte[24:]])

product = product_region_area[:]
reactant = reactant_region_area[:]

total = product + reactant #+ unknown
yield_prod = product/ total
yield_react = reactant/ total
#yield_unknown = unknown/ total

# %%
data_102728 = pd.DataFrame({
    'model ID': df_summary['Model ID'].to_numpy(),
    'time': time,
    'temp': temp,
    'sulf': sulfonating_agent,
    'anly': analyte,
    'yield product': yield_prod,

})
data_102728.head(21)


# %%
df_save = data_102728

#%%
list(df_save.columns)
# %%
df_save.to_csv('extracted_data_round_redo.csv', index=False)

# %% [markdown]
'''
> ### Data for 102728 HPLC Data
'''

