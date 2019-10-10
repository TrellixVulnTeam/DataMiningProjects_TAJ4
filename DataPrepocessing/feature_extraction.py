import numpy as np
from sklearn import preprocessing
from sklearn.decomposition import PCA
from DataPrepocessing import converting_discrete_attributes_to_continuous as prepo

## PCA
dataoutp = np.delete(prepo.dataout, [1,6], axis=1)
datapca = PCA()
datapca.fit(dataoutp)
print(datapca.explained_variance_ratio_)
print(datapca.explained_variance_)
print(datapca.n_components_)
print(datapca.singular_values_)

## a good number of components
dataoutp = preprocessing.Normalizer().fit_transform(dataoutp)
datapca = PCA(n_components=3)
datapca.fit(dataoutp)
print(datapca.explained_variance_ratio_)


