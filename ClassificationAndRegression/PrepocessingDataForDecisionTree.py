from sklearn import preprocessing
from sklearn.datasets import load_iris
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.impute import SimpleImputer
import numpy as np
import collections

## prepocessing for data attributes
incomedata = pd.read_csv('incomedata/census-income.data',header=None)
Y = incomedata.to_numpy()
output = np.empty([199523,1])
for i in range(40):
    if(i == 24):
        continue
    target = Y[:,i]
    target = target.reshape(-1,1)
    X = SimpleImputer(missing_values='?', strategy='most_frequent').fit_transform(target)
    X = SimpleImputer(missing_values=' ?', strategy='most_frequent').fit_transform(target)
    if(i in (0,5,16,17,18,38)):
        output = np.hstack((output, X))
        continue
    if(i in (24,25,26,28)):
        continue
    le = preprocessing.OneHotEncoder()
    X = le.fit_transform(target)
    X = X.toarray()
    X = X.astype(int)
    output = np.hstack((output,X))
output = np.delete(output,0,axis=1)
print(np.size(output,0))
print(np.size(output,1))
np.savetxt('incomedata/DecisionTreeData',output,fmt='%s')