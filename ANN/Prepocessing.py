import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
from sklearn.preprocessing import OneHotEncoder, LabelEncoder, KBinsDiscretizer
from sklearn.impute import SimpleImputer

os.chdir('C://Users//Beichen//PycharmProjects//CS548//ANN')

data = pd.read_csv('census-income.csv')
print(data.shape)
data = data.drop(columns='instance weight')

# Data Exploration
print(data.shape)

# Missing value percentage:
miss_count = (data == ' ?').sum(axis=0)
miss_percent = miss_count / data.shape[0]
print(miss_percent)

# Regression target 'age' distribution
plt.hist(data['age'], color='blue', edgecolor='black', bins=30)
plt.title('Histogram of AGE')
plt.xlabel('age')
plt.ylabel('frequency')
plt.show()

# Classification target 'income' distribution
income_minus = (data['income_50k'] == '-50000').sum()
print(income_minus / (data.shape[0]))
plt.hist(data['income_50k'], color='blue', edgecolor='black')
plt.title('Histogram of income')
plt.ylabel('frequency')
plt.show()

# Missing value handling
# Drop attributes: ‘Mig_chg_msa’/ ‘Mig_chg_reg’/ ‘Mig_mov_reg’/ ‘Mig_prev_sunbelt’
data1 = data.drop(columns=['mig_chg_msa', 'mig_chg_reg', 'mig_mov_reg', 'mig_prev_sunbelt'])
data1.shape  # 199523, 37

# Use most frequent value in the attribute to replace the missing value inside:
# ‘State_prev_res’, ‘country_father’, ‘country_mother’, ‘country_self’
data1.loc[data1['state_prev_res'] == ' ?', 'state_prev_res'] = np.nan
data1.loc[data1['country_father'] == ' ?', 'country_father'] = np.nan
data1.loc[data1['country_mother'] == ' ?', 'country_mother'] = np.nan
data1.loc[data1['country_self'] == ' ?', 'country_self'] = np.nan

imp_mf = SimpleImputer(missing_values=np.nan, strategy='most_frequent')
state_prev_res = imp_mf.fit_transform(data1['state_prev_res'].to_frame())

data1['state_prev_res'] = state_prev_res
data1['country_father'] = imp_mf.fit_transform(data1['country_father'].to_frame())
data1['country_mother'] = imp_mf.fit_transform(data1['country_mother'].to_frame())
data1['country_self'] = imp_mf.fit_transform(data1['country_self'].to_frame())

data1.shape  # 191640, 37  around 3.9% data are dropped
data2 = data1
data2.to_csv('data2.csv', index=False)

# Percentage for each class after dropping na data
income_minus2 = (data2['income_50k'] == '-50000').sum()
print(income_minus2 / (data2.shape[0]))  # 0.938

# Prepare data for classification
# one-hot encoding for the categorical attributes
colnames = data2.columns
continous_names = ['age', 'wage_per_hour', 'capital_gains', 'capital_losses', 'stock_dividends', 'num_emp',
                   'weeks_worked', 'income_50k']
cat_names = list(set(data2.columns) - set(continous_names))
cat_index = [j for j in range(data2.shape[1]) if colnames[j] in cat_names]

onehot = OneHotEncoder()
X_2 = data2.iloc[:, cat_index]
onehot.fit(X_2)
onehot_X2 = onehot.transform(X_2).toarray()

data3 = data2.drop(columns=cat_names)  # 199523, 8
# data3 = data3.reset_index(drop=True)
data3 = pd.concat([pd.DataFrame(onehot_X2), data3], axis=1)

data3.shape  # 199523, 474

data3.to_csv('data3.csv', index=False)

# Prepare data for linear regression
data4 = pd.read_csv('data2.csv')
colnames = data4.columns
continous_names = ['age', 'wage_per_hour', 'capital_gains', 'capital_losses', 'stock_dividends', 'num_emp',
                   'weeks_worked', 'income_50k']
cat_names = list(set(data4.columns) - set(continous_names))
cat_index = [j for j in range(data4.shape[1]) if colnames[j] in cat_names]

enc = LabelEncoder()
for i in cat_index:
    data4.iloc[:, i] = enc.fit_transform(data4.iloc[:, i])
data4.to_csv('data4.csv', index=False)

# Prepare data for oneR
data5 = pd.read_csv('data4.csv')
continous_names = ['age', 'wage_per_hour', 'capital_gains', 'capital_losses', 'stock_dividends', 'num_emp',
                   'weeks_worked']
for i in continous_names:
    data5[i] = KBinsDiscretizer(n_bins=3, encode='ordinal', strategy='uniform').fit_transform(data5[i].to_frame())

data5.to_csv('data5.csv', index=False)

# Prepare data for MLP_clas on age
data2 = pd.read_csv('data2.csv')

onehot = OneHotEncoder()
X_2 = data2.loc[:, 'sex']+data2.loc[:, 'education']
X_2 = X_2.to_numpy().reshape(-1, 1)


Y_2 = data2.loc[:, 'income_50k']
Y_2 = Y_2.to_numpy().reshape(-1, 1)

onehot.fit(X_2)
onehot_X2 = onehot.transform(X_2).toarray()

lebel_en = LabelEncoder()
lab_Y2 = lebel_en.fit(Y_2).transform(Y_2)


data3 = data2.drop(columns=cat_names)  # 199523, 8
# data3 = data3.reset_index(drop=True)
data3 = pd.concat([pd.DataFrame(onehot_X2), data3], axis=1)

data3.shape  # 199523, 474

data3.to_csv('data3.csv', index=False)