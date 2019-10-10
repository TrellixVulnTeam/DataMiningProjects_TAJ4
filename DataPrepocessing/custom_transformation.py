import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import FunctionTransformer

raw_data = pd.read_csv('flag_data/flag.data')
areadata = raw_data['4']
areadata = np.array(areadata).reshape(-1,1)

## transform this attribute from "thousands of square kms" to "thousands of square miles"
print(areadata)
areadata_trans = FunctionTransformer(lambda x: x/1.6/1.6).transform(areadata)
print(areadata_trans)
plt.hist(areadata,100)
plt.show()




