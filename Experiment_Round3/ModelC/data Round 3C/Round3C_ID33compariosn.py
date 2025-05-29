
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
#df_summary = pd.read_excel("/Users/clarat/Documents/Sun_Lab/PNNL/2002 Design Summary Sheet.xlsx") #2002 summary
df_summary = pd.read_excel("../Iteration3_Summary.xlsx", sheet_name="Iteration3C")
vial_id_33 = df_summary[df_summary['Vial ID'] == 33]
vial_id_33

# Get the CSV files only
data33 = pd.read_csv("../data Round 3C/102622 (33).csv")
data33
plt.plot(data33['X33'], data33['Y33'])
plt.show()

data33_new = pd.read_excel("../data Round 3C/3C-102622_HT-HJ-E wells.xlsx", sheet_name="Sheet1", header=None)
data33_new.columns = ['X33', 'Y33']
data33_new
plt.plot(data33_new.iloc[:, 0], data33_new.iloc[:, 1])
plt.xlabel('X33')
plt.ylabel('Y33')
plt.title('Plot of data33_new')
plt.show()

# %%
def fit_peak_range(df,n,min, max,filter):
    peak_list = []
    for i in range(df.shape[1]//2):
        chrom = Chromatogram(df, cols={'time':f'X{i+n}', 'signal':f'Y{i+n}'},time_window=[min,max])
        chrom.correct_baseline()
        peak_list.append(chrom.fit_peaks(prominence=filter))
    return peak_list

list_product_first = fit_peak_range(df=data33,n=33,min=0.0, max =1.0, filter=0.2)
list_product_second = fit_peak_range(df=data33_new,n= 33, min=1.0, max=2.0, filter =0.2)
list_product = list_product_first + list_product_second

# %%
list_reactant_first = fit_peak_range(df=data33,n=33, min=3.0, max=4.0, filter=0.2)
list_reactant_second = fit_peak_range(df=data33_new,n=33, min=2.0, max=3.0, filter =0.3)
list_reactant = list_reactant_first +list_reactant_second
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
# time = np.array([element for element in  df_summary['time']])
# time = time[:45] #np.hstack([time[:21],time[24:]])
time = vial_id_33['time']

# temp = np.array([element for element in  df_summary['temp']])
# temp = temp[:45] #np.hstack([temp[:21],temp[24:]])
temp = vial_id_33['temp']
# sulfonating_agent= np.array([element for element in  df_summary['Sulfonating Agent']])
# sulfonating_agent= sulfonating_agent[:45] #np.hstack([sulfonating_agent[:21],sulfonating_agent[24:]])
sulfonating_agent = vial_id_33['Sulfonating Agent']
# analyte = np.array([element for element in  df_summary['Analyte']])
# analyte= analyte[:45] #np.hstack([analyte[:21],analyte[24:]])
analyte = vial_id_33['Analyte']
# product = product_region_area[:45] #np.hstack([product_area[:21],product_area[24:]])
product = product_region_area
reactant = reactant_region_area
# reactant = reactant_region_area[:45] #np.hstack([reactant_area[:21],reactant_area[24:]])
# #unknown = np.hstack([unknown_area[:21],unknown_area[24:]])


total = product + reactant #+ unknown
yield_prod = product/ total
yield_react = reactant/ total

print(yield_prod)

# %%
label_x= ['time','sulfonating agent','analyte','temp']
x_data = [time,sulfonating_agent,analyte,temp]
y_data = [yield_prod]#[product,reactant,product/reactant,unknown]
line = ['.-','.-','.-','.']
label_y = ['product']#['product','reactant','ratio','unknown'] 



#pair[pair[:,0].argsort()]
# %%
data_102622 = pd.DataFrame({
    'time': [time,time],
    'temp': [temp,temp],
    'sulf': [sulfonating_agent,sulfonating_agent],
    'analyte': [analyte,analyte],
    'area product': product,
    'area reactant': reactant,
#    'area unknown': unknown,
    'yield product': yield_prod,
    'yield reactant': yield_react,
    #'yield unknown': yield_unknown
})
data_102622