import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.impute import SimpleImputer

incomedata = pd.read_csv('incomedata/census-income.data',header=None)
sex = incomedata.to_numpy()[:,12]
sex.astype(str)
print(sex)

citizenship = pd.read_csv('incomedata/census-income.data',header=None)
citizenship = citizenship.to_numpy()[:,34]
citizenship = citizenship.reshape(-1,1)
citizenship = SimpleImputer(missing_values=' ?', strategy='most_frequent').fit_transform(citizenship)
citizenship = citizenship.flatten()
print(citizenship)

eduction = pd.read_csv('incomedata/census-income.data',header=None)
eduction = eduction.to_numpy()[:,4]
eduction = eduction.reshape(-1,1)
eduction = SimpleImputer(missing_values=' ?', strategy='most_frequent').fit_transform(eduction)
eduction = eduction.flatten()
print(eduction)

income = pd.read_csv('incomedata/income50k',header=None)
income = np.array(income).flatten()
income = income.astype(int)
print(income)


result = {}
for i in range(eduction.size):
    result[eduction[i] + str(income[i])] = result.get(eduction[i] + str(income[i]),0)+1
#x = ['Female < 50k','Male < 50k','Male > 50k','Female > 50k']
output = {}
for key,value in result.items():
    for key2 , value2 in result.items():
        if(key[:-1] == key2[:-1]):
            output[key[:-1]] = output.get(key[:-1],0) + value/value2
print(output)
plt.bar(range(len(output)), list(output.values()), width=0.5, align='center')
plt.xticks(range(len(output)), list(output.keys()))
plt.show()
#plt.bar(income,sex)
#plt.show()