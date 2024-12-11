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

import fnmatch
import re
# %%
#df_summary = pd.read_excel("/Users/clarat/Documents/Sun_Lab/PNNL/2002 Design Summary Sheet.xlsx") #2002 summary
df_summary = pd.read_excel("/Users/ctuwsunlab/Documents/GitHub/PNNL-ML_for_Organic_Flow_Battery_Materials/Round3/ModelC/Iteration3_Summary.xlsx", sheet_name="Iteration3C")
df_summary.head()
# %%
# Get the CSV files only
directory = "/Users/ctuwsunlab/Documents/GitHub/PNNL-ML_for_Organic_Flow_Battery_Materials/Round3/ModelC/data Round 3C/"
files = fnmatch.filter(os.listdir(directory), '*.csv')

# Sort files based on the second number in the filename
sorted_files = np.array(sorted(files, key=lambda s: int(re.findall(r'\d+', s)[1]) if len(re.findall(r'\d+', s)) >= 2 else 0))
sorted_files = sorted_files[1:len(sorted_files)-2]

#%%
#read them into pandas
columns_list = [pd.read_csv(directory+files,skiprows=0, sep=',').columns.to_numpy() for files in sorted_files]

# %%
df_list = [pd.read_csv(directory+ sorted_files[i],skiprows=1, sep=',', names=[columns_list[i][0],columns_list[i][1]]).dropna(axis=1, how='all') for i in range(len(columns_list[:32]))]
df_32 = pd.concat(df_list,axis=1)
# %%
# Normalize only the Y columns in df_32
norm_df_32 = df_32.copy()
y_columns = [col for col in df_32.columns if col.startswith('Y')]
norm_df_32[y_columns] = (df_32[y_columns] - df_32[y_columns].mean()) / (df_32[y_columns].max() - df_32[y_columns].min())

# 33 to 48 
df_48 = pd.read_csv("/Users/ctuwsunlab/Documents/GitHub/PNNL-ML_for_Organic_Flow_Battery_Materials/Round3/ModelC/data Round 3C/3C_spec33_48.csv")
# Normalize only the Y columns in df_48
norm_df_48 = df_48.copy()
y_columns_48 = [col for col in df_48.columns if col.startswith('Y')]
norm_df_48[y_columns_48] = (df_48[y_columns_48] - df_48[y_columns_48].mean()) / (df_48[y_columns_48].max() - df_48[y_columns_48].min())


#%%
fig, axs = plt.subplots(nrows=8, ncols=6, figsize=(15, 20))

for i in range(32):
    row = i // 6
    col = i % 6
    axs[row, col].plot(df_32[f'X{i+1}'], df_32[f'Y{i+1}'])
    axs[row, col].set_title(f'Plot of X{i+1} vs Y{i+1}')
    axs[row, col].set_xlabel(f'X{i+1}')
    axs[row, col].set_ylabel(f'Y{i+1}')
    axs[row, col].grid()

for i in range(16):
    row = (i + 32) // 6
    col = (i + 32) % 6
    axs[row, col].plot(df_48[f'X{i+33}'], df_48[f'Y{i+33}'])
    axs[row, col].set_title(f'Plot of X{i+33} vs Y{i+33}')
    axs[row, col].set_xlabel(f'X{i+33}')
    axs[row, col].set_ylabel(f'Y{i+33}')
    axs[row, col].grid()

plt.tight_layout()
plt.show()

# %%
def fit_peak_range(df,i, min, max,filter):
    peak_list = []
    for i in range(i, i+df.shape[1]//2):
        chrom = Chromatogram(df, cols={'time':f'X{i+1}', 'signal':f'Y{i+1}'},time_window=[min,max])
        
        chrom.correct_baseline()

        peak_list.append(chrom.fit_peaks(prominence=filter))

    return peak_list


# %%
list_product_first = fit_peak_range(df=df_32,i=0,min=0.0, max =1.0, filter=0.9)
list_product_second = fit_peak_range(df=df_48,i = 32, min=1.0, max=2.0, filter =0.9)
list_product = list_product_first +list_product_second
norm_product = fit_peak_range(df=norm_df_32,i=0,min=0.0, max =1.0, filter=0.9) + fit_peak_range(df=norm_df_48,i = 32, min=1.0, max=2.0, filter =0.9)
list_product
# %%
list_reactant_first = fit_peak_range(df=df_32,i=0, min=3.0, max=4.0, filter=0.2)
list_reactant_second = fit_peak_range(df=df_48,i = 32, min=2.0, max=3.0, filter =0.3)
list_reactant = list_reactant_first +list_reactant_second
norm_reactant = fit_peak_range(df=norm_df_32,i=0, min=3.0, max=4.0, filter=0.2) + fit_peak_range(df=norm_df_48,i = 32, min=2.0, max=3.0, filter =0.3)
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
product_region_area = area_select(list_product)
reactant_region_area = area_select(list_reactant)

norm_area_product = area_select(norm_product)
norm_area_reactant = area_select(norm_reactant)


# %%
time = np.array([element for element in  df_summary['time']])
time = time[:45] #np.hstack([time[:21],time[24:]])

temp = np.array([element for element in  df_summary['temp']])
temp = temp[:45] #np.hstack([temp[:21],temp[24:]])

sulfonating_agent= np.array([element for element in  df_summary['Sulfonating Agent']])
sulfonating_agent= sulfonating_agent[:45] #np.hstack([sulfonating_agent[:21],sulfonating_agent[24:]])

analyte = np.array([element for element in  df_summary['Analyte']])
analyte= analyte[:45] #np.hstack([analyte[:21],analyte[24:]])

product = product_region_area[:45] #np.hstack([product_area[:21],product_area[24:]])
reactant = reactant_region_area[:45] #np.hstack([reactant_area[:21],reactant_area[24:]])


total = product + reactant #+ unknown
yield_prod = product/ total
yield_react = reactant/ total


# %%
data_102622 = pd.DataFrame({
    '3C_time': time,
    '3C_temp': temp,
    '3C_sulf': sulfonating_agent,
    '3C_anly': analyte,
    '3C_yield product': yield_prod,

})
data_102622.head(20)

df_save = data_102622
list(df_save.columns)
# %%
df_save.to_csv('/Users/ctuwsunlab/Documents/GitHub/PNNL-ML_for_Organic_Flow_Battery_Materials/Round3/ModelC/Extracted_data_round3C.csv', index=False)

# %% [markdown]
'''
> ### HPLC data extracted
'''

# %%
