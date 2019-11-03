import os

os.chdir('D:\Courses\WPI-ds\CS548\Project3')

import numpy as np
import pandas as pd
from sklearn.model_selection import cross_val_score
from sklearn import metrics
from time import perf_counter
from sklearn.model_selection import KFold
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor, plot_tree
from scipy.stats import pearsonr
from Project2 import modeltree
from Project2.linear_regr import linear_regr
from sklearn.tree import export_graphviz

from sklearn.neural_network import MLPRegressor


# Prepare feature attributes for regression tree and model tree and MLP_reg
data3 = pd.read_csv('data3.csv')
feature_cols1 = data3.drop(columns='age').columns[0:-1]
X1 = data3[feature_cols1]
X1 = X1.to_numpy()

y1 = data3['age']
y1 = np.array(y1, dtype=np.int)

# Prepare feature attributes for linear regression
data4 = pd.read_csv('data4.csv')
feature_cols2 = data4.columns[1:-1]
X2 = data4[feature_cols2]
X2 = X2.to_numpy()

y2 = data4['age']
y2 = np.array(y2, dtype=np.int)

'''Guiding Questions'''
# [10 points] Three Guiding Questions for the Regression Experiments: (at most 1/3 page)
# 1.   	Does occupation have influence on age?
# 2.   	Is there a significant difference on age between races?
# 3.   	How parents’ marriage status affects the individual’s age?
import seaborn as sns
# 1.
data2 = pd.read_csv('data2.csv')
ax = sns.boxplot(x="major_occ_code", y="age", data=data2).set_title('Box plot of age versus major occupation code')
# 2.
ax2 = sns.boxplot(x="race", y="age", data=data2).set_title('Box plot of age versus race')
# 3.
ax3 = sns.boxplot(x="marital_stat", y="age", data=data2).set_title('Box plot of age versus marital status')

'''ZeroR'''
kf = KFold(n_splits=10, shuffle=True, random_state=10)
mse_zeroR = []
cocoe = []
for train_index, test_index in kf.split(X2, y2):
    y_train, y_test = y2[train_index], y2[test_index]
    y_pred = [y_train.mean()] * len(y_test)
    #mse_zeroR.append(metrics.mean_squared_error(y_pred, y_test))
    y_pred = np.array(y_pred, dtype=np.int)
    y_test = np.array(y_test)
    cocoe.append(pearsonr(y_pred, y_test))

sum(mse_zeroR)/len(mse_zeroR)

t0 = perf_counter()
pred_y = y2.mean()
t1 = perf_counter()
round(t1 - t0,10)

'''Linear Regression'''
# 10 fold cross validation with random sampling
kf = KFold(n_splits=10, shuffle=True, random_state=10)
lm = LinearRegression(normalize=True)
# MSE
#mse_score = metrics.make_scorer(metrics.mean_squared_error)
#t0 = perf_counter()
#mse_result = cross_val_score(lm, X2, y2, cv=kf, scoring=mse_socre)
#t1 = perf_counter()

cc_lm = np.array([])
mse_lm = np.array([])
for train_index, test_index in kf.split(X2, y2):
    X_train, X_test, y_train, y_test = X2[train_index], X2[test_index], y2[train_index], y2[test_index]
    lm.fit(X_train, y_train)
    y_pred = lm.predict(X_test)
    mse_score = np.append(mse_lm, metrics.mean_squared_error(y_pred, y_test))
    cc_lm = np.append(cc_lm, pearsonr(y_test, y_pred)[0])
# AVG_mse
mse_lm.mean()
# AVG_cc
cc_lm.mean()

# Final model
t0 = perf_counter()
lm_final = lm.fit(X2, y2)
t1 = perf_counter()


# Time spent
round(t1 - t0, 2)

'''Regression Tree'''
# 10 fold cross validation with random sampling
kf = KFold(n_splits=10, shuffle=True, random_state=10)
rt = DecisionTreeRegressor(random_state=1, max_depth=20)

cc_rt = np.array([])
mse_rt = np.array([])
for train_index, test_index in kf.split(X1, y1):
    X_train, X_test, y_train, y_test = X1[train_index], X1[test_index], y1[train_index], y1[test_index]
    rt.fit(X_train, y_train)
    y_pred = rt.predict(X_test)
    mse_rt = np.append(mse_rt, metrics.mean_squared_error(y_pred, y_test))
    cc_rt = np.append(cc_rt, pearsonr(y_test, y_pred)[0])
# AVG_mse
mse_rt.mean()
# AVG_cc
cc_rt.mean()

# Final model
t0 = perf_counter()
rt_final = rt.fit(X1, y1)
t1 = perf_counter()

import graphviz
#plot_tree(rt.fit(X1, y1))
dot_data = export_graphviz(rt_final)
graph = graphviz.Source(dot_data)
print(graph)


rt_final.get_depth() #111   # 20
rt_final.get_n_leaves() #112122  # 17148

# Time spent
round(t1 - t0, 2)

'''Model Tree'''

model1 = linear_regr()
mt = modeltree.ModelTree(model=model1, max_depth=3, min_samples_leaf=50, search_type='adaptive')

cc_mt = np.array([])
mse_mt = np.array([])
t0 = perf_counter()
for train_index, test_index in kf.split(X2, y2):
    X_train, X_test, y_train, y_test = X2[train_index], X2[test_index], y2[train_index], y2[test_index]
    mt.fit(X_train, y_train)
    y_pred = mt.predict(X_test)
    mse_mt = np.append(mse_mt, metrics.mean_squared_error(y_test, y_pred))
    cc_mt = np.append(cc_mt, pearsonr(y_test, y_pred)[0])
t1 = perf_counter()

# AVG_mse
mse_mt.mean()
# AVG_cc
cc_mt.mean()

# Final model
t0 = perf_counter()
mt_final2 = mt.fit(X=X2, y=y2, verbose=True)
t1 = perf_counter()

# Time spent
round(t1 - t0, 2)

'''MLP Regression'''
kf = KFold(n_splits=10, shuffle=True, random_state=10)
MLP_reg = MLPRegressor(hidden_layer_sizes=(30,30),
                                     activation='relu',
                                     solver='adam',
                                     learning_rate='adaptive',
                                     max_iter=100,
                                     learning_rate_init=0.01,
                                     alpha=0.01)
mse_s = metrics.make_scorer(metrics.mean_squared_error)



t0 = perf_counter()
mse_result1 = cross_val_score(MLP_reg, X1, y1, cv=kf, scoring= mse_s)
t1 = perf_counter()



cc_rt = np.array([])
mse_rt = np.array([])
for train_index, test_index in kf.split(X1, y1):
    X_train, X_test, y_train, y_test = X1[train_index], X1[test_index], y1[train_index], y1[test_index]
    rt.fit(X_train, y_train)
    y_pred = rt.predict(X_test)
    mse_rt = np.append(mse_rt, metrics.mean_squared_error(y_pred, y_test))
    cc_rt = np.append(cc_rt, pearsonr(y_test, y_pred)[0])
# AVG_mse
mse_rt.mean()
# AVG_cc
cc_rt.mean()

# Final model
t0 = perf_counter()
rt_final = rt.fit(X1, y1)
t1 = perf_counter()