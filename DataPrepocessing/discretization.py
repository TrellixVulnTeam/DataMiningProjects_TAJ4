import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import preprocessing


raw_data = pd.read_csv('flag_data/flag.data')
popudata = raw_data['5']
plt.hist(popudata,100)
plt.show()
popudata = np.array(popudata).reshape(-1,1)

## K-bins discretization
popudata_uni = preprocessing.KBinsDiscretizer(n_bins=3, encode='ordinal', strategy='uniform').fit_transform(popudata)
plt.hist(popudata_uni,100)
plt.show()
popudata_quan = preprocessing.KBinsDiscretizer(n_bins=3, encode='ordinal', strategy='quantile').fit_transform(popudata)
plt.hist(popudata_quan,100)
plt.show()
popudata_kmeans = preprocessing.KBinsDiscretizer(n_bins=3, encode='ordinal', strategy='kmeans').fit_transform(popudata)
plt.hist(popudata_kmeans,100)
plt.show()

## Feature binarization
popudata_bina = preprocessing.Binarizer(threshold=50).fit_transform(popudata)
plt.hist(popudata_bina,100)
plt.show()