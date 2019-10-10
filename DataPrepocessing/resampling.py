from sklearn import utils
from DataPrepocessing import converting_discrete_attributes_to_continuous as prepo

# without replacement using uniform distribution
data_ru = utils.resample(prepo.dataout,replace=False,n_samples=116)
print(data_ru)

# random sampling with replacement using uniform distribution
data_noru = utils.resample(prepo.dataout,replace=True,n_samples=116)
print(data_noru)

# stratified random sampling without replacement
data_rsr = utils.resample(prepo.dataout,replace=True,stratify=prepo.dataout[:,2])
