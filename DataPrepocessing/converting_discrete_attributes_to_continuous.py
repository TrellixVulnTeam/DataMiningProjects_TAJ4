import pandas as pd
import numpy as np
from sklearn import preprocessing

rawdata = pd.read_csv('flag_data/flag.data')
## encoding to data1
data = rawdata[['2','3','6','7','18','29','30']]
data = np.array(data)
enc = preprocessing.OneHotEncoder()
enc.fit(data)
data1 = enc.transform(data).toarray()
## write into the rawdata
raw_data = np.array(rawdata)
dataout = np.delete(raw_data, [0,1,2,5,6,17,28,29], axis=1)
dataout = np.hstack((dataout,data1))




