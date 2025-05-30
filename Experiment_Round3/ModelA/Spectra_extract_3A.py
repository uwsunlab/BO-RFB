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
df_summary = pd.read_excel("Iteration3_Summary.xlsx", sheet_name="Iteration3A")
df_summary

#%%
big_df = pd.read_csv("data Round 3A /3A_combined.csv")
big_df

norm_df = big_df.copy()
y_columns = [col for col in big_df.columns if col.startswith('Y')]
norm_df[y_columns] = (big_df[y_columns] - big_df[y_columns].mean()) / (big_df[y_columns].max() - big_df[y_columns].min())

#%%
plt.plot(big_df['X31'], big_df['Y31'])
plt.plot(big_df['X32'], big_df['Y32'])
plt.plot(big_df['X32'], big_df['Y33'])
plt.xlabel('X')
plt.ylabel('Y')
plt.title('70 - Plot of 2A: 31,32,33')
plt.legend(['Y31','Y32','Y33'])
plt.show()

# %%
fig, axs = plt.subplots(nrows=8, ncols=6, figsize=(15, 20))

for i in range(48):
    row = i // 6
    col = i % 6
    axs[row, col].plot(big_df[f'X{i+1}'], big_df[f'Y{i+1}'])
    axs[row, col].set_title(f'Plot of X{i+1} vs Y{i+1}')
    axs[row, col].set_xlabel(f'X{i+1}')
    axs[row, col].set_ylabel(f'Y{i+1}')
    plt.grid()
plt.tight_layout()
plt.show()


# %%
def fit_peak_range(df,min, max,filter):
    peak_list = []
    for i in range(df.shape[1]//2):
        chrom = Chromatogram(df, cols={'time':f'X{i+1}', 'signal':f'Y{i+1}'},time_window=[min,max])
        
        chrom.correct_baseline()

        peak_list.append(chrom.fit_peaks(prominence=filter))

    return peak_list

# %%
# %%
list_product = fit_peak_range(big_df,1, 2.0, 0.9)
#list_product = fit_peak_range(0.0, 0.6, 0.2)
list_product
# %%
#list_reactant = fit_peak_range(3.8, 5.0, 0.5)
list_reactant = fit_peak_range(big_df,2.1, 3.2, 0.3)
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


#%%
reactant_region_area = area_select(list_reactant)
product_region_area = area_select(list_product)

# %%
time = np.array([element for element in  df_summary['time']])
time = time[:45] #np.hstack([time[:21],time[24:]])

temp = np.array([element for element in  df_summary['temp']])
temp = temp[:45] #np.hstack([temp[:21],temp[24:]])

sulfonating_agent= np.array([element for element in  df_summary['Sulfonating Agent']])
sulfonating_agent= sulfonating_agent[:45] #np.hstack([sulfonating_agent[:21],sulfonating_agent[24:]])

analyte = np.array([element for element in  df_summary['Analyte']])
analyte= analyte[:45] #np.hstack([analyte[:21],analyte[24:]])

product = product_region_area[:45]
reactant = reactant_region_area[:45]

total = product + reactant #+ unknown
yield_prod = product/ total
yield_react = reactant/ total

# %%
data_102612 = pd.DataFrame({
    '3A_time': time,
    '3A_temp': temp,
    '3A_sulf': sulfonating_agent,
    '3A_anly': analyte,
    '3A_yield product': yield_prod,

})

# %%
df_save = data_102612
list(df_save.columns)
# %%
df_save.to_csv('Extracted_data_round3A.csv', index=False)

# %% [markdown]
'''
> ### Data for 102119 HPLC Data
'''
