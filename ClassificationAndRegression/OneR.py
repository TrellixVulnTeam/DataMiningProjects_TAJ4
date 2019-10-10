import pandas as pd
import numpy as np
from collections import Counter
from sklearn import preprocessing
from sklearn import metrics
from sklearn.model_selection import StratifiedKFold

## get the data
incomedata = pd.read_csv('incomedata/income50k',header=None)
target = incomedata.to_numpy()

X = pd.read_csv('incomedata/OneRData',sep=' ',header=None)
X = X.to_numpy()

## oneR
skf = StratifiedKFold(n_splits=10, random_state=1)
m = X.shape[1]
roc_oneR = np.array([])
for train_index, test_index in skf.split(X, target):
	X_train, X_test, y_train, y_test = X[train_index], X[test_index], target[train_index], target[test_index]
	roc_record_train = np.array([])
	y_pred_train = y_train.copy()
	y_pred = y_test.copy()
	for j in range(m):
            classes = list(set(X_train[:, j]))
    	for i in classes:
        	index = X_train[:, j] == i
        	y_pred_train[index] = Counter(y_train[index]).most_common()[0][0]
    	roc_record_train = np.append(roc_record_train, metrics.roc_auc_score(y_train, y_pred_train))
	col_ind = np.argmax(roc_record_train)
	classes = list(set(X_train[:, col_ind]))
	dic_ = {}
	for i in classes:
    	    index = X_train[:, col_ind] == i
    	dic_[i] = Counter(y_train[index]).most_common()[0][0]
	classes = list(set(X_test[:, col_ind]))
	for i in classes:
    	    index = X_test[:, col_ind] == i
    	y_pred[index] = dic_[i]
	roc_oneR = np.append(roc_oneR, metrics.roc_auc_score(y_test, y_pred))



