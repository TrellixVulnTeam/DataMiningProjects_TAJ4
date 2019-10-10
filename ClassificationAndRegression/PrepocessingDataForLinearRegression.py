import pandas as pd
import numpy as np
from sklearn import tree
from sklearn.impute import SimpleImputer
from sklearn import preprocessing
from sklearn.linear_model import LinearRegression

incomedata = pd.read_csv('incomedata/census-income.test',header=None)
X = incomedata.to_numpy()
incomedata_np = incomedata.to_numpy()
X = X[:,0:41]
## Get norminal data
X = np.delete(X,[0,5,16,17,18,29,38],axis=1)
## Missing values
X = SimpleImputer(missing_values='?', strategy='most_frequent').fit_transform(X)
## Convert to continuous int values

##enc = preprocessing.OneHotEncoder()
##X = enc.fit_transform(X)
enc = preprocessing.LabelEncoder()
for i in range(34):
    X[:,i] = enc.fit_transform(X[:,i])
#X = X.toarray()
X = X.astype(int)
## Missing values and Convert to int values
Y = incomedata_np[:,[0,5,16,17,18,29,38]]
Y = SimpleImputer(missing_values='?', strategy='most_frequent').fit_transform(Y)
Y = Y.astype(int)
## Get the X, Y for regression
X = np.hstack((X,Y))
## Y = incomedata[:,0]
## save to text
np.savetxt('incomedata/LRtestData',X)

########################################################################################################################

incomedata = pd.read_csv('incomedata/census-income.data',header=None)
X = incomedata.to_numpy()
incomedata_np = incomedata.to_numpy()
X = X[:,0:41]
## Get norminal data
X = np.delete(X,[0,5,16,17,18,29,38],axis=1)
## Missing values
X = SimpleImputer(missing_values='?', strategy='most_frequent').fit_transform(X)
## Convert to continuous int values

#enc = preprocessing.OneHotEncoder()
#X = enc.fit_transform(X)
enc = preprocessing.LabelEncoder()
for i in range(7):
    X[:,i] = enc.fit_transform(X[:,i])
#X = X.toarray()
X = X.astype(int)
## Missing values and Convert to int values
Y = incomedata_np[:,[0,5,16,17,18,29,38]]
Y = SimpleImputer(missing_values='?', strategy='most_frequent').fit_transform(Y)
Y = Y.astype(int)
## Get the X, Y for regression
X = np.hstack((X,Y))
#X = np.delete(X,0,axis=1)
#Y = incomedata[:,0]
np.savetxt('incomedata/LRData',X)