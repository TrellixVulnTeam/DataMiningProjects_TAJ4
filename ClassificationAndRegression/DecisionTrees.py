import pandas as pd
import numpy as np
from sklearn import tree
from sklearn import preprocessing
import time
from sklearn.model_selection import KFold
from sklearn.metrics import roc_auc_score

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
clf = tree.DecisionTreeClassifier()
clf = clf.fit(X,target)
depth = 0
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

    depth += clf.get_depth()
    score += roc_auc_score(test_check, test_target)
    print(score)

end = time.time()
print(score/10)
print(end - start)
print(depth/10)





