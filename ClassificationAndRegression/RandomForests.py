import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import KFold
import time


start = time.time()

## get the incometarget
incomedata = pd.read_csv('incomedata/census-income.data',header=None)
incomedata = incomedata.to_numpy()
target = incomedata[:,41]
enc = preprocessing.LabelEncoder()
target = enc.fit_transform(target)
target = np.array(target, dtype=int)

X = pd.read_csv('incomedata/DecisionTreeData',sep=' ',header = None)
X = X.to_numpy()

## construct the tree
clf = RandomForestClassifier(n_estimators=10)
clf = clf.fit(X,target)

## test
score = 0
kf = KFold(n_splits=10)
print(X.shape)
for trainindex, testindex in kf.split(X):
    test = X[testindex,:]
    test_check = target[testindex]
    train = X[trainindex,:]
    train_target = target[trainindex]

    clf.fit(train, train_target)
    test_target = clf.predict(test)

    score += roc_auc_score(test_check, test_target)
    print(score)

end = time.time()
print(score/10)
print(end - start)