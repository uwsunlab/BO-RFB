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
df_summary = pd.read_excel("2303 Experiment Details.xlsx", sheet_name="summary")
df_summary.head()

# Get the CSV files only
directory = "240725_102404_Test2A/"
files = fnmatch.filter(os.listdir(directory), '*.csv')

sorted_files = np.array(sorted(files, key=lambda s: int(re.findall(r'\d+', s)[1]) if len(re.findall(r'\d+', s)) >= 2 else 0))
sorted_files

# %%
#read them into pandas
columns_list = [pd.read_csv(directory+files,skiprows=0, sep=',').columns.to_numpy() for files in sorted_files]

df_list = [pd.read_csv(directory+ sorted_files[i],skiprows=1, sep=',', names=[columns_list[i][0],columns_list[i][1]]).dropna(axis=1, how='all') for i in range(len(sorted_files))]
df_list
# %%
big_df = pd.concat(df_list,axis=1)
big_df

# %%
def plot_spec(id):
  data = [big_df[columns_list[id][0]],big_df[columns_list[id][1]]]
  #norm_y = (big_df['Y%s'%sorted_files[id]] - big_df['Y%s'%sorted_files[id]].min()) / (big_df['Y%s'%sorted_files[id]].max() - big_df['Y%s'%sorted_files[id]].min())
  peaks, _ = find_peaks(data[1], prominence=0.005)
  col = (np.random.random(), np.random.random(), np.random.random())
  plt.plot(data[0],data[1])#,c=col)
  #plt.plot(data[0],norm_y)#,c=col)
  plt.vlines(data[0].values[peaks], 0, np.max(data[1]), linestyle='--', color='tab:grey')#'dodgerblue'
  #plt.title('%s'%sorted_files[id])
  plt.title('%s'%columns_list[id])#len(peaks))
# %%
#plot the normalized spectra
fig = plt.figure(figsize=(20,25))
#Spectra vs Weights
for i in range(0,len(df_list)):
  plt.subplot(1+len(df_list)//6,6,i+1)
  plot_spec(i)
plt.tight_layout()

# %%
def fit_peak_range(min, max,filter):
    peak_list = []
    for i in range(big_df.shape[1]//2):
        chrom = Chromatogram(big_df, cols={'time':columns_list[i][0], 'signal':columns_list[i][1]},time_window=[min,max])
        chrom.correct_baseline()
        peak_list.append(chrom.fit_peaks(prominence=filter))

    return peak_list

#%% [markdown]
'''
# NOTE
- Product = 0.29 min RT (between 0:1)
- Reactant + unknown = 3.92 min RT  (between 3.5:5.0) 
- Sulfuric Acid = 5.25 min RT  (between 5:6)
\
\
All other peaks are potential side products \
Area counts for compounds can assume intersection at origin and 8910360 = 0.053 g/ml

'''
# %%
list_product = fit_peak_range(0, 0.5, 0.5)
list_reactant = fit_peak_range(3.8, 5.0, 0.5)


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

def sum_area(peak_list):
    array = np.zeros(len(peak_list))
    for i in range(len(peak_list)): 
        for j in range(len(peak_list[i])):
            val = peak_list[i]['area'].sum()
        array[i] = val
    return array
#%%
reactant_region_area = sum_area(list_reactant)
product_region_area = sum_area(list_product)

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

total = product + reactant 
yield_prod = product/ total
yield_react = reactant/ total

# %%
data_102404 = pd.DataFrame({
    '2A_time': time,
    '2A_temp': temp,
    '2A_sulf': sulfonating_agent,
    '2A_anly': analyte,
    '2A_yield product': yield_prod,

})

df_save = data_102404
#%%
list(df_save.columns)
# %%
df_save.to_csv('extracted_data_round2A.csv', index=False)
