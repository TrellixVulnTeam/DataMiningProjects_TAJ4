import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.impute import SimpleImputer
from sklearn.impute import IterativeImputer
from DataPrepocessing import converting_discrete_attributes_to_continuous as prepo

raw_data = pd.read_csv('flag_data/flag.data')
## 1
areadata = raw_data['4']
areadata = np.array(areadata).reshape(-1,1)
mean = SimpleImputer(missing_values=0, strategy='mean').fit_transform(areadata)
median = SimpleImputer(missing_values=0, strategy='median').fit_transform(areadata)
most_freq = SimpleImputer(missing_values=0, strategy='most_frequent').fit_transform(areadata)
constant = SimpleImputer(missing_values=0, strategy='constant').fit_transform(areadata)


plt.hist(areadata,100)
plt.show()
plt.subplot(221)
plt.hist(mean,100)
plt.subplot(222)
plt.hist(median,100)
plt.subplot(223)
plt.hist(most_freq,100)
plt.subplot(224)
plt.hist(constant,100)
plt.show()


## 2
imp = IterativeImputer(missing_values=np.nan)
plt.hist(imp.fit_transform(prepo.dataout))
plt.show()













