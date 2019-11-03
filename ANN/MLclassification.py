import numpy as np
import pandas as pd

from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn import metrics
from time import perf_counter
from sklearn import preprocessing
from sklearn.ensemble import RandomForestClassifier
import os
from collections import Counter
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import average_precision_score
from sklearn.model_selection import StratifiedKFold


os.chdir('D:\Courses\WPI-ds\CS548\Project3')

data3 = pd.read_csv('data3.csv')

# Prepare feature attributes
feature_cols = data3.columns[:-1]
X = data3[feature_cols]
X = X.to_numpy()

# Prepare target attribute
y = data3['income_50k']
enc = preprocessing.LabelEncoder()
y = enc.fit_transform(y)
y = np.array(y, dtype=np.int)

# Prepare feature attributes for OneR
data5 = pd.read_csv('data5.csv')
feature_cols2 = data5.columns[0:-1]
X5 = data5[feature_cols2]
X5 = X5.to_numpy()
y5 = data5.iloc[:,-1]
enc = preprocessing.LabelEncoder()
y5 = enc.fit_transform(y5)
y5 = np.array(y5, dtype=np.int)

'''OneR'''

skf = StratifiedKFold(n_splits=2, random_state=1)
m = X5.shape[1]
roc_oneR = np.array([])
for train_index, test_index in skf.split(X5, y5):
    X_train, X_test, y_train, y_test = X5[train_index], X5[test_index], y5[train_index], y5[test_index]
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



'''DecisionTree'''
# 10 fold cross validation with stratified sampling
skf = StratifiedKFold(n_splits=2, random_state=1)
# skf_index = skf.split(X, y)
clf = DecisionTreeClassifier(criterion='entropy')
roc_score = metrics.make_scorer(metrics.roc_auc_score)

t0 = perf_counter()
roc_result2 = cross_val_score(clf, X, y, cv=skf, scoring= roc_score)
t1 = perf_counter()

# Test result
roc_result2.mean()

# Final model
t0 = perf_counter()
clf_final = clf.fit(X, y)
t1 = perf_counter()

# Visualize the tree
plot_tree(clf_final)


clf_final.get_depth() #83   # 20
clf_final.get_n_leaves() #10372  # 4118

# Time spent
round(t1-t0, 2)

'''RandomForest'''
rf_clf = RandomForestClassifier(criterion='entropy', random_state=1, n_estimators=20)
t0 = perf_counter()
roc_result_rf = cross_val_score(rf_clf, X, y, cv=skf, scoring=roc_score)
t1 = perf_counter()
# Test result
roc_result_rf.mean()
# Time spent
round(t1-t0, 2)
# Final model
t0 = perf_counter()
rf_clf_final = rf_clf.fit(X, y)
t1 = perf_counter()

'''MLP Classifier'''
# 10 fold cross validation with stratified sampling
skf = StratifiedKFold(n_splits=10, random_state=1)

MLP_clf = MLPClassifier(hidden_layer_sizes=(20, 20))
roc_score = metrics.make_scorer(metrics.roc_auc_score)
avg_prec_score = metrics.make_scorer(metrics.average_precision_score)
recall_score = metrics.make_scorer(metrics.recall_score)
accuracy_score = metrics.make_scorer(metrics.accuracy_score)
t0 = perf_counter()
roc_result_MLP = cross_val_score(MLP_clf, onehot_X2, lab_Y2, cv=skf, scoring=roc_score)
roc_result_MLP_precision = cross_val_score(MLP_clf, onehot_X2, lab_Y2, cv=skf, scoring=avg_prec_score)
roc_result_MLP_recall = cross_val_score(MLP_clf, onehot_X2, lab_Y2, cv=skf, scoring=recall_score)
roc_result_MLP_accuracy = cross_val_score(MLP_clf, onehot_X2, lab_Y2, cv=skf, scoring=accuracy_score)
t1 = perf_counter()

# Time spent
round(t1-t0, 2)
# Final model
t0 = perf_counter()
rf_clf_final = MLP_clf.fit(X, y)
t1 = perf_counter()

roc_result_rf.mean()




#clf.fit(X, target)
# Test
accu = []
t0 = perf_counter()
for train_index, test_index in skf_index:
    X_train = X[train_index]
    X_test = X[test_index]
    y_train = y[train_index]
    y_test = y[test_index]
    # Train Decision Tree classifier
    clf = clf.fit(X_train, y_train)
    # Predict y_test
    y_pred = clf.predict(X_test)
    # Record metrics
    accu.append(metrics.roc_auc_score(y_test, y_pred))
t1 = perf_counter()

# Time spent
round(t1-t0, 2)





score = metrics.make_scorer(metrics.precision_score)
cross_val_score(clf, X, y, cv=skf, scoring= roc_score)

# Time spent







for train_index, test_index in skfq.split(q, p):
    print("TRAIN:", train_index, "TEST:", test_index)
    X_train, X_test, y_train, y_test  = q[train_index], q[test_index], p[train_index], p[test_index]
    print(X_train, X_test, y_train, y_test)

roc = metrics.make_scorer(metrics.roc_auc_score())

cross_val_score(clf, )