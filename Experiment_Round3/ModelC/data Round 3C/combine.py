#%%
import numpy as np
import pandas as pd
import os

# %%
path = '../data Round 3C/'
excel_files = [f for f in os.listdir(path) if f.endswith('.xlsx')]
excel_files = sorted(excel_files)
excel_files
# %%
df_list = []
for file in excel_files:
    # Read all sheets in the Excel file
    sheets = pd.read_excel(os.path.join(path, file), sheet_name=None)
    # Iterate over each sheet and add to the DataFrame list
    for sheet_name, df in sheets.items():
        df_list.append(df)

# Concatenate all DataFrames
final_df = pd.concat(df_list, axis=1, ignore_index=True)

print(final_df)

# %%
# Assign unique X and Y labels for each column pair
num_columns = final_df.shape[1]
new_columns = []

for i in range(num_columns // 2):
    new_columns.append(f'X{i+1}')
    new_columns.append(f'Y{i+1}')

# If there are an odd number of columns, add the last one as 'X' with the next index
if num_columns % 2 != 0:
    new_columns.append(f'X{num_columns // 2 + 1}')

final_df.columns = new_columns

print(final_df)
# %%
# Save the final DataFrame as a CSV file
output_path = os.path.join(path, '3C_combined.csv')
final_df.to_csv(output_path, index=False)
# %%
