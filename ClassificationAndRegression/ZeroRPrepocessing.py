from sklearn import preprocessing
from sklearn.datasets import load_iris
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.impute import SimpleImputer
import numpy as np
import collections

## prepocessing for data attribute Income
incomedata = pd.read_csv('incomedata/census-income.data',header=None)
X = incomedata.to_numpy()
target = incomedata.to_numpy()[:,41]
target = target.reshape(-1,1)
target = SimpleImputer(missing_values='?', strategy='most_frequent').fit_transform(target)
le = preprocessing.LabelEncoder()
X = le.fit_transform(target)
X = X.astype(int)
X = X.reshape(-1,1)
np.savetxt('incomedata/income50k',X)



