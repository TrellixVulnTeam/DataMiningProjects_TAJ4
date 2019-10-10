import pandas as pd
import numpy as np
import time

start = time.time()
incomedata = pd.read_csv('incomedata/DecisionTreeData',sep=' ',header=None)
X = incomedata.to_numpy()
target = X[:, 0]
end = time.time()
print(end - start)
print(np.average(target))
print(np.var(target))