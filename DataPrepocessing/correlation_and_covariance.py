import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from DataPrepocessing import converting_discrete_attributes_to_continuous as prepo

## correlation
corr = np.corrcoef(prepo.dataout.T.astype(float))
print(corr)
map = sns.heatmap(corr)
map.plot()
plt.show()
np.savetxt('matrix',corr)

## covariance
cova = np.cov(prepo.dataout.T.astype(float))
print(cova)
map = sns.heatmap(cova)
map.plot()
plt.show()
np.savetxt('matrix',cova)
































