import pandas as pd
import numpy as np
from sklearn.feature_selection import VarianceThreshold
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.feature_selection import f_classif
from sklearn.feature_selection import mutual_info_classif
from sklearn.feature_selection import f_regression
from sklearn.svm import SVR
from sklearn.feature_selection import RFE
from DataPrepocessing import converting_discrete_attributes_to_continuous as prepo

rawdata = pd.read_csv('flag_data/flag.data')
dataoutc = np.delete(prepo.dataout, 6, axis=1)
dataoutr = np.delete(prepo.dataout, 1, axis=1)
data7 = np.array(rawdata)[:,6]
data5 = prepo.dataout[:,1]

## variance threshold
datavs = VarianceThreshold(threshold=3).fit_transform(prepo.dataout)

## selectKBest
# classification
print(SelectKBest(chi2,k=50).fit_transform(dataoutc,data7.astype('float')))
print(SelectKBest(f_classif,k=50).fit_transform(dataoutc,data7.astype('float')))
print(SelectKBest(mutual_info_classif,k=50).fit_transform(dataoutc,data7.astype('float')))
# regression
print(SelectKBest(f_regression,k=50).fit_transform(dataoutr,data5.astype('float')))
print(SelectKBest(mutual_info_classif,k=50).fit_transform(dataoutr,data5.astype('float')))
# RFE
estimator = SVR(kernel="linear")
selector = RFE(estimator, 20, step=1).fit_transform(dataoutr,data5)


















