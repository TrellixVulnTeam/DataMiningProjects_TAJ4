from sklearn import preprocessing
import numpy as np
X = [[ 1., -1.,  2.],[ 2.,  0.,  0.],[ 0.,  1., -1.]]

## scale
X_scaled = preprocessing.scale(X)
X_scaled.mean(axis= 0)
X_scaled.std(axis=0)
# or
scaler = preprocessing.StandardScaler()
scaler.fit_transform(X)

## scaling features to a range
# [0,1]
min_max_scaler = preprocessing.MinMaxScaler()
X_minmax = min_max_scaler.fit_transform(X)
# [-1,1]
max_abs_scaler = preprocessing.MaxAbsScaler()
X_maxabs = max_abs_scaler.fit_transform(X)

## normalization
X_normalized = preprocessing.normalize(X, norm = 'l2')

## non-linear transformation
# mapping to a gaussian distribution
pt = preprocessing.PowerTransformer(method='box-cox', standardize=False)
X_lognormal = np.random.RandomState(6).lognormal(size=(3, 3))
# mapping to a uniform distribution
qt = preprocessing.QuantileTransformer(random_state=0)
X_trans  = qt.fit_transform(X)
np.percentile(X[:, 0], [0, 25, 50, 75, 100])

## discretization
# feature binarization
binarizer = preprocessing.Binarizer(threshold=1.0).fix(X)
# K-bins discretization
est = preprocessing.KBinsDiscretizer(n_bins=[3, 2, 2], encode='ordinal').fix(X)
est.transform(X)

## encoding categorical features
genders = ['female', 'male']
locations = ['from Africa', 'from Asia', 'from Europe', 'from US']
browsers = ['uses Chrome', 'uses Firefox', 'uses IE', 'uses Safari']
X = [['male', 'from US', 'uses Safari'], ['female', 'from Europe', 'uses Firefox']]
enc = preprocessing.OneHotEncoder(categories=[genders, locations, browsers])
print(enc.fit(X))
print(enc.transform([['female', 'from Europe', 'uses Firefox']]).toarray())

## generating polynomial features
from sklearn.preprocessing import PolynomialFeatures
X = np.arange(9).reshape(3, 3)
poly0 = PolynomialFeatures(2)
poly1 = PolynomialFeatures(degree=3, interaction_only=True)
print(X)
print(poly0.fit_transform(X))
print(poly1.fit_transform(X))

## custom transformers
from  sklearn.preprocessing import FunctionTransformer
transformer = FunctionTransformer(np.log1p, validate=True)
X = np.array([[0, 1],[2, 3]])
print(transformer.transform(X))




