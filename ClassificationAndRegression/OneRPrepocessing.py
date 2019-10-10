from sklearn import preprocessing
import pandas as pd
from sklearn.impute import SimpleImputer
import numpy as np


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
    if (i in (24, 25, 26, 28)):
        continue
    if(i in (5,16,17,18,38)):
        X = preprocessing.KBinsDiscretizer(n_bins=3, encode='ordinal', strategy='uniform').fit_transform(X)
        output = np.hstack((output, X))
    output = np.hstack((output, X))
    print(i)

output = np.delete(output,0,axis=1)
np.savetxt('incomedata/OneRData',output,fmt='%s')


