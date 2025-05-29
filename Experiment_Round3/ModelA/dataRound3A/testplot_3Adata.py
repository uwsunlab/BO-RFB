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

fig, axs = plt.subplots(2, 4, figsize=(15, 8))  # Create a 2x4 grid of subplots

for i in range(1, 9):
    data = pd.read_excel("../dataRound3A/3A-102612_HT-HJ-F wells.xlsx", sheet_name="Sheet%s" % i)
    row = (i - 1) // 4
    col = (i - 1) % 4
    axs[row, col].plot(data.to_numpy()[:, 0], data.to_numpy()[:, 1])
    axs[row, col].set_title('data%s' % (40 + i))

plt.tight_layout()  # Adjust subplots to fit into figure area.
plt.show()



# %%
