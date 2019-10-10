import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
import time


start = time.time()
## get the data
incomedata = pd.read_csv('incomedata/DecisionTreeData',sep=' ',header=None)
X = incomedata.to_numpy()
target = X[:,0]
np.delete(X, 0, axis=1)


## fit the model
reg = LinearRegression(normalize=True)

## get the error
error = 0
kf = KFold(n_splits=10)
print(X.shape)
for trainindex, testindex in kf.split(X):
    test = X[testindex,:]
    test_check = target[testindex]
    train = X[trainindex,:]
    train_target = target[trainindex]

    reg.fit(train, train_target)
    test_target = reg.predict(test)
    error += mean_squared_error(test_check, test_target)
    print(error)
end = time.time()
print(error/10)
print(end - start)