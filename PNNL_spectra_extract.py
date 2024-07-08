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
df_summary = pd.read_excel("/Users/clarat/Documents/Sun_Lab/PNNL/208590_ESMI Synthesis EXPERIMENT LOG6a37cb242d3e0142ead426d85c590b82b20ece59623cbc8e445d91ce28a08e63.xlsx",sheet_name='2006')
df_summary.head()
# %%
import fnmatch
import re

# get the CSV files only
# Get the CSV files only
files = fnmatch.filter(os.listdir('/Users/clarat/Documents/Sun_Lab/PNNL/102119 HPLC Data'), '*.csv')
#files = fnmatch.filter(os.listdir('/Users/clarat/Documents/Sun_Lab/PNNL/102107 UV Spectra'), '*.csv')

# Sort files based on the second number in the filename
sorted_files = sorted(files, key=lambda s: int(re.findall(r'\d+', s)[1]) if len(re.findall(r'\d+', s)) >= 2 else 0)

print(sorted_files)

# %%
df = pd.read_csv('/Users/clarat/Documents/Sun_Lab/PNNL/102119 HPLC Data/102119 (2).csv',skiprows=1, sep='\t').columns.item()
df

# %%
#read them into pandas
columns_list = [pd.read_csv('/Users/clarat/Documents/Sun_Lab/PNNL/102119 HPLC Data/'+files,skiprows=1, sep='\t').columns.item() for files in sorted_files]

df_list = [pd.read_csv('/Users/clarat/Documents/Sun_Lab/PNNL/102119 HPLC Data/'+ sorted_files[i],skiprows=2, sep='\t',names=['X%s'%columns_list[i],'Y%s'%columns_list[i]]).dropna(axis=1, how='all') for i in range(len(sorted_files))]
# df_columns = [df_list[i].columns.item() for i in range(len(df_list))]

# #

# len(df_list)

# df_list
df_list

# %%
big_df = pd.concat(df_list,axis=1)
big_df
# %%
Chromatogram(big_df, cols={'time':'X%s'%columns_list[0], 'signal':'Y%s'%columns_list[0]}).show()
plt.plot(big_df['X102119_02'],big_df['Y102119_02'])
plt.plot(big_df['X102119_03'],big_df['Y102119_03'])
# %%
def plot_spec(id):
  data = [big_df['X%s'%columns_list[id]],big_df['Y%s'%columns_list[id]]]
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
# %%
def fit_peak_range(min, max,filter):
    peak_list = []
    for i in range(big_df.shape[1]//2):
        #print(i)
        #signal_norm = (big_df['Y%s'%sorted_files[i]] - big_df['Y%s'%sorted_files[i]].min()) / (big_df['Y%s'%sorted_files[i]].max() - big_df['Y%s'%sorted_files[i]].min())
        chrom = Chromatogram(big_df, cols={'time':'X%s'%columns_list[i], 'signal':'Y%s'%columns_list[i]},time_window=[min,max])
        chrom.correct_baseline()
        #print(i)
        peak_list.append(chrom.fit_peaks(prominence=filter))
        # if i ==21:
        #     peak_list.append(chrom.fit_peaks(prominence=filter))
        # # elif i  == 31:
        # #     peak_list.append(chrom.fit_peaks(prominence=0.06))
        # else: 
        #     peak_list.append(chrom.fit_peaks())
    return peak_list
# %%
list_all = fit_peak_range(0, 7, 0.02)
list_all
# %%
list_product = fit_peak_range(0, 0.5, 0.05)
# %%
list_reactant = fit_peak_range(3.5, 4, 0.2)


#%% [markdown]
'''
# NOTE
- Product = 0.29 min RT
- Reactant = 3.92 min RT
- Sulfuric Acid = 5.25 min RT 
\
\
All other peaks are potential side products \
Area counts for compounds can assume intersection at origin and 8910360 = 0.053 g/ml

'''

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

#acid_peak = select_peaks_area(5.2)[3:]
product_area = select_peaks_area(0.2,list_product)
acid_area = select_peaks_area(5.2,list_all)
reactant_area = select_peaks_area(3.9,list_reactant )+select_peaks_area(3.8,list_reactant )
unknown_area = select_peaks_area(4.5,list_all)+select_peaks_area(4.4,list_all)
acid_area ,product_area, reactant_area
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
unknown = np.hstack([unknown_area[:21],unknown_area[24:]])


total = product + reactant + unknown
yield_prod = product/ total
yield_react = reactant/ total
yield_unknown = unknown/ total
# %%
#plt.plot(acid_area)
# plt.plot(product_area,'.')
# plt.plot(reactant_area,'.')
# Plot the data with different colors for every three points
np.random.seed(41)
for i in range(0, len(product_area), 3):
    color1 = plt.cm.spring(i / len(product_area))  # Generate a color from the colormap
    color2 =  plt.cm.winter(i / len(reactant_area))  # Generate a color from the colormap
    #color = np.random.rand(3,) #random color
    plt.plot(range(i, i+3),product_area[i:i+3], color='tab:blue',marker='o', linestyle='')
    plt.plot(range(i, i+3),reactant_area[i:i+3], color='tab:orange',marker='o', linestyle='')
    plt.vlines(i+2.5, 0, 2.3, linestyle='--', color='tab:grey')#'dodgerblue'
    plt.fill_betweenx(np.linspace(0, 2.3, 100),21-0.5,24-0.5,color='tab:grey', alpha=0.01)
    #plt.plot(range(i, i+3),reactant_area[i:i+3], color=color1,marker='o', linestyle='')
    #plt.plot(range(i, i+3),(product_area/reactant_area)[i:i+3], color=color1,marker='o', linestyle='')
    #plt.plot(range(i, i+3),unknown_area[i:i+3], color=color1,marker='o', linestyle='')

plt.title('Visualization of spread between conditions')
plt.xlabel('samples')
plt.ylabel('output')
plt.legend(['product','reactant'])

# %%

# time = np.array([element for element in  df_summary['Sample Time (min)']])[:48]
# temp = np.array([element for element in  df_summary['Temperature (degC)']])[:48]
# sulfinating_agent= np.array([element for element in  df_summary['Sulfonating Agent\n(wt%)']])[:48]
# analyte = np.array([element for element in  df_summary['Reagent Ratio\n(mg/mL reagent/sulfonating agent)']])[:48]
# time_52.sort()
# time_02.sort()
label_x= ['time','sulfonating agent','analyte','temp']
x_data = [time,sulfonating_agent,analyte,temp]
y_data = [yield_prod,yield_unknown]#[product,reactant,product/reactant,unknown]
line = ['.-','.-','.-','.']

label_y = ['product','unknown']#['product','reactant','ratio','unknown'] 
count = 0

fig = plt.figure(figsize=(20,10))
for i in range(len(y_data)):
    for j in range(len(x_data)):
        count +=1
        pair = np.array([x_data[j],y_data[i]]).transpose()
        pair = pair[pair[:,0].argsort()]
        plt.subplot(len(y_data),len(x_data),count)
        #plt.plot(x_data[j].reshape(-1,3).mean(axis=1),area.reshape(-1,3).mean(axis=1),'.-')
        #plt.plot(pair[:,0].reshape(-1,3).mean(axis=1)[1:],pair[:,1].reshape(-1,3).mean(axis=1)[1:],'.-')
        #plt.plot(pair[:,0].reshape(-1,3).mean(axis=1)[1:],pair[:,1].reshape(-1,3).max(axis=1)[1:],'.-')
        plt.plot(pair[:,0],pair[:,1],line[j])
        if label_y[i] != 'unknown':
            plt.axhline(y=0.95, color='tab:green',linestyle = '--' ,linewidth=2)
        plt.xlabel(label_x[j])
        plt.ylabel('Yield of  %s'%label_y[i])

#pair[pair[:,0].argsort()]
# %%
data_102119 = pd.DataFrame({
    'time': time,
    'temp': temp,
    'sulf': sulfonating_agent,
    'analyte': analyte,
    'area product': product,
    'area reactant': reactant,
    'area unknown': unknown,
    'yield product': yield_prod,
    'yield reactant': yield_react,
    'yield unknown': yield_unknown
})
data_102119.sort_values('temp').head(11)
# %%
def surface_all(ax,data_temp,input1,input2,output,color):
        # Grab some test data.
    # X, Y, Z = axes3d.get_test_data(0.05)
    X = data_temp.sort_values(input1)[input1].to_numpy()
    Y = data_temp.sort_values(input2)[input2].to_numpy()
    X, Y = np.meshgrid(X, Y)
    out1,out2 = data_temp.sort_values(input1)[output].to_numpy(),data_temp.sort_values(input2)[output].to_numpy() #data_temp[output].to_numpy(),data_temp[output].to_numpy()#
    Z = np.matmul(out1.reshape(len(out1),1),out2.reshape(1,len(out2)))
    #Z1,Z2 = np.meshgrid(data_temp.sort_values(input1)[output].to_numpy(), data_temp.sort_values(input2)[output].to_numpy())

    ## Plot outside 
    # fig = plt.figure()
    # ax = fig.add_subplot(projection='3d')
    ## Plot a basic wireframe.
    #ax.plot_wireframe(X, Y, Z1, rstride=10, cstride=0,color='tab:grey')
    ax.plot_surface(X, Y, Z,alpha= 0.3,color=color)
 
    #ax.view_init(elev=20, azim=120) 
    ax.set_xlabel(input1)
    ax.set_ylabel(input2)
    ax.set_zlabel('output')
#%%
def contour_all(ax,data_temp,input1,input2,output,color):
    # Grab some test data.
    # X, Y, Z = axes3d.get_test_data(0.05)
    X = data_temp.sort_values(input1)[input1].to_numpy()
    Y = data_temp.sort_values(input2)[input2].to_numpy()
    X, Y = np.meshgrid(X, Y) 
    out1,out2 = data_temp.sort_values(input1)[output].to_numpy(),data_temp.sort_values(input2)[output].to_numpy() #data_temp[output].to_numpy(),data_temp[output].to_numpy()#
    Z = np.matmul(out1.reshape(len(out1),1),out2.reshape(1,len(out2)))
    #Z1,Z2 = np.meshgrid(data_temp.sort_values(input1)[output].to_numpy(), data_temp.sort_values(input2)[output].to_numpy())

    ## Plot outside 
    # fig = plt.figure()
    # ax = fig.add_subplot(projection='3d')
    ## Plot a basic wireframe.
    #ax.plot_wireframe(X, Y, Z1, rstride=10, cstride=0,color='tab:grey')
    ax.contour(X, Y, Z,8,alpha= 0.3,colors=color)
    #ax.contourf(X, Y, Z, 8, alpha=.5) # plt.contourf(X, Y, f(X, Y), 8, alpha=.75, cmap=plt.cm.hot)
    #C = ax.contour(X, Y, Z, 8, colors='black', linewidth=.25,linstlyle = '--') # C = plt.contour(X, Y, f(X,Y), 8, colors='black', linewidth=.5)
    #ax.clabel(C, inline=1, fontsize=1)


    #ax.view_init(elev=20, azim=120) 
    ax.set_xlabel(input1)
    ax.set_ylabel(input2)

# %%
# Surface plots
fig,(ax,ax1,ax2) = plt.subplots(1, 3, figsize=(20, 8), subplot_kw={'projection': '3d'})

# fig = plt.figure(figsize=(15, 8))
# ax = fig.add_subplot(projection='3d')
surface_all(ax,data_102119,'time','sulf','yield product','tab:blue')
#ax.scatter(data['time'], data['sulf'], data['Output'], c='b', marker='o')  # 'c' is the color and 'marker' defines the shape of markers
ax.set_title('time vs sulfonating agent')

#ax1 = fig.add_subplot(projection='3d')
surface_all(ax1,data_102119,'time','analyte','yield product','tab:orange')
#ax1.scatter(data['time'], data['analyte'], data['Output'], c='r', marker='o')  # 'c' is the color and 'marker' defines the shape of markers
ax1.set_title('time vs analyte')

#ax2 = fig.add_subplot(projection='3d')
surface_all(ax2,data_102119,'sulf','analyte','yield product','tab:green')
#ax2.scatter(data['sulf'], data['analyte'], data['Output'], c='g', marker='o')  # 'c' is the color and 'marker' defines the shape of markers
ax2.set_title('sulfonating agent vs analyte')

# %%
fig,(ax,ax1,ax2) = plt.subplots(1, 3, figsize=(15, 5))

#contour_all(ax,avg_data,'time',"sulf",'yield product','k')
contour_all(ax,data_102119,'time','sulf','yield product','tab:blue')
contour_all(ax1,data_102119,'time','analyte','yield product','tab:orange')
contour_all(ax2,data_102119,'sulf','analyte','yield product','tab:green')
# %%
fig,(ax,ax1,ax2) = plt.subplots(1, 3, figsize=(15, 5))
contour_all(ax,data_102119,'temp','time','yield product','tab:blue')
contour_all(ax1,data_102119,'temp','sulf','yield product','tab:orange')
contour_all(ax2,data_102119,'temp','analyte','yield product','tab:green')

# %%
