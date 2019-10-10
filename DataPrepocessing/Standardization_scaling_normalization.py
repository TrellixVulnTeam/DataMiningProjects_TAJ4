import numpy as np
import pandas as pd
from sklearn import preprocessing
import matplotlib.pyplot as plt

raw_data = pd.read_csv('flag_data/flag.data')
areadata = raw_data['4']
areadata = np.array(areadata).reshape(-1,1)

## standarization
areadata_scaled = preprocessing.scale(areadata)
plt.hist(areadata_scaled,100)
plt.show()

## scale to a range
areadata_minmax = preprocessing.MinMaxScaler().fit_transform(areadata)
plt.hist(areadata_minmax,100)
plt.show()

areadata_maxabs = preprocessing.MaxAbsScaler().fit_transform(areadata)
plt.hist(areadata_maxabs,100)
plt.show()

areadata_robust_scale = preprocessing.robust_scale(areadata)
plt.hist(areadata_robust_scale,100)
plt.show()

areadata_robust = preprocessing.RobustScaler().fit_transform(areadata)
plt.hist(areadata_robust,100)
plt.show()

## mapping
# uniform distribution
areadata_quantile0 = preprocessing.quantile_transform(areadata,axis=0,n_quantiles=194)
plt.hist(areadata_quantile0,100)
plt.show()

areadata_quantile1 = preprocessing.QuantileTransformer(random_state=0).fit_transform(areadata)
plt.hist(areadata_quantile1,100)
plt.show()

# gaussian distribution
areadata_normal1 = preprocessing.QuantileTransformer(output_distribution='normal', random_state=0).fit_transform(areadata)
plt.hist(areadata_normal1,100)
plt.show()

areadata_normal2 = preprocessing.PowerTransformer(method='yeo-johnson', standardize=True, copy=True).fit_transform(areadata)
plt.hist(areadata_normal2,100)
plt.show()
# normalization
areadata_norm = preprocessing.Normalizer().fit_transform(areadata)
plt.hist(areadata_norm,100)
plt.show()


