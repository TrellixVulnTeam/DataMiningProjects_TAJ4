import pandas as pd
import collections
import numpy as np
import time
from sklearn.metrics import roc_auc_score
## count numbers of male and female in Attribute Sex

start = time.time()
Y = pd.read_csv('incomedata/income50k',header = None)
Y = Y.to_numpy()
Y = Y.astype(int)
Y = Y.flatten()
predict = np.zeros((199523,1),dtype=np.int)
print(predict)
end = time.time()
print(roc_auc_score(Y,predict))
print(collections.Counter(Y))
print(end - start)