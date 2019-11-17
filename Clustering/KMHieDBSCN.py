import pandas as pd
import time
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import numpy as np
##get the data dividens_from_stock, weeks_work_in_a_year, wages_per_hour
data = pd.read_csv('C://Users\Beichen\PycharmProjects\CS548\Clustering\census-income.csv',usecols=[4],header=None)
data.to_csv('C://Users\Beichen\PycharmProjects\CS548\Clustering\KmeansData1',index=False)
data1 = pd.read_csv('C://Users\Beichen\PycharmProjects\CS548\Clustering\KmeansData1')

data2 = pd.read_csv('C://Users\Beichen\PycharmProjects\CS548\Clustering\KmeansData1')
data1 = data1.to_numpy()

from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
data1 = le.fit_transform(data1)
data1 = data1[1:,:]
data1 = data1.astype(int)
data2 = data2.to_numpy()
data2 = data2[1:,:]
data2 = data2.astype(int)
np.random.seed(1)
sample_ind = np.random.choice(data.shape[0], size=1000, replace=False)

## Kmeans
kmeans = KMeans(n_clusters=3,random_state=1).fit(data1[0:1000,:])
data1 = data1[0:1000]

##Calculate SSE
sse = []
for k in range(1, 20):
    kmeans = KMeans(n_clusters=k,random_state=1).fit(data1[0:1000,:])
    sse.append(kmeans.inertia_)
plt.plot(sse)
plt.show()

## heatmap
from sklearn.cluster import DBSCAN
dbs = DBSCAN().fit(data1)
from sklearn.metrics import pairwise_distances
distmatrix = pairwise_distances(data1)
distmatrix = distmatrix[dbs.labels_.argsort()]
labels2 = kmeans.labels_.copy()
labels2.sort()

kmeansinci = DBSCAN().fit(data1)
incidencematrix = np.zeros((1000,1000))
for i in range(1000):
    for j in range(1000):
        if labels2[i] == labels2[j]:
            incidencematrix[i, j] = 1


corrmatrix = np.corrcoef(distmatrix,incidencematrix)
import seaborn as sns
sns.heatmap(corrmatrix)

np.random.seed(1)
sample_ind = np.random.choice(data1.shape[0], size=1000, replace=False)
data1 = data1[sample_ind,:]
## Silhouette Coefficient
from sklearn import metrics
kmeans_model = KMeans(n_clusters=3,random_state=1).fit(data1.reshape(-1,1))
labels = kmeans_model.labels_
Score = metrics.silhouette_score(data1[0:1000,:],labels)


## MDS
from sklearn.manifold import MDS
mds = MDS(2, max_iter=100, n_init=1)
Y = mds.fit_transform(data1.reshape(-1,1))
plt.scatter(Y[:,0],Y[:,1],c=kmeans_model.fit_predict(data1.reshape(-1,1)))

## TSNE
from sklearn.manifold import TSNE
ts = TSNE(n_components=2).fit_transform(data1)
plt.scatter(ts[:,0],ts[:,1],c=kmeans_model.fit_predict(data1))
plt.show()

## percent
labels = kmeans.labels_
percent_clas = []
for i in range(kmeans.n_clusters):
    percent_clas.append(sum(labels == i)/len(labels))
print(percent_clas)




## Hierachical Clustering
from sklearn import preprocessing
data1 = preprocessing.normalize(data1)

from sklearn.cluster import AgglomerativeClustering
import time
start = time.time()
hpre = AgglomerativeClustering(n_clusters=3).fit_predict(data1)
end = time.time()
print(end - start)
hfit = AgglomerativeClustering(n_clusters=3).fit(data1[0:1000,:])

from sklearn.manifold import TSNE
ts = TSNE(n_components=2).fit_transform(data1)
plt.scatter(ts[:,0],ts[:,1],c=hpre)
plt.show()

from sklearn.manifold import MDS
mds = MDS(2, max_iter=100, n_init=1)
Y = mds.fit_transform(data1[1:1001,:])
plt.scatter(Y[:,0],Y[:,1],c=hpre)


labels = hfit.labels_
percent_clas = []
for i in range(hfit.n_clusters_):
    percent_clas.append(sum(labels == i)/len(labels))
print(percent_clas)


## Silhouette Coefficient
from sklearn import metrics
kmeans_model = KMeans(n_clusters=2,random_state=1).fit(data1[0:1000,:])
labels = hfit.labels_
Score = metrics.silhouette_score(data1[0:1000,:],labels)

##from scipy.cluster.hierarchy import dendrogram
##model = AgglomerativeClustering().fit((data1[:,0],data1[:,2]))
import seaborn as sns
sns.clustermap(data1[1:1000,:])
##https://scikit-learn.org/dev/auto_examples/cluster/plot_agglomerative_dendrogram.html

from sklearn import metrics
labels_true = np.ones(1000)

print(metrics.homogeneity_completeness_v_measure(labels_true,labels))

## KNN plot
from sklearn import preprocessing
data2 = preprocessing.normalize(data2[sample_ind,:])
from sklearn.neighbors import NearestNeighbors
knn = NearestNeighbors(n_neighbors=5, n_jobs=-1).fit(data2)
knn_dis = knn.kneighbors()[0][:, -1]
knn_dis.sort()

plt.figure()
plt.plot(range(1,1001), knn_dis, marker='o')
plt.xlabel("Points sorted according to distance")
plt.ylabel("5th nearest Neighbor distance")
plt.title('GQ1: DBSCAN n_neighbors=5')

## DBSCAN
DB = DBSCAN(eps=0.40, min_samples=6, n_jobs=-1).fit(data2[sample_ind,:])

labels_db = DB.labels_

cl_percent = np.unique(labels_db, return_counts=True)[1]/1000
cl_percent

from sklearn.cluster import DBSCAN
start = time.time()
dbsc = DBSCAN().fit(data2)
db_pred  = DBSCAN().fit_predict(data2)
end = time.time()
print(end - start)

from sklearn.manifold import MDS
##mds = MDS(2, max_iter=100, n_init=1)
from sklearn.manifold  import TSNE
Y = TSNE(n_components=2).fit_transform(data2)
plt.scatter(Y[:,0],Y[:,1],c=db_pred)

labels = dbsc.labels_
percent_clas = []
for i in range(7):
    percent_clas.append(sum(labels == i)/len(labels))
print(percent_clas)

unique_value = []
for x in dbsc.labels_:
    if x not in unique_value:
        unique_value.append(x)

from sklearn import metrics
Score = metrics.silhouette_score(data1[0:1000, :], labels)

from sklearn.metrics import pairwise_distances
distmatrix = pairwise_distances(db_pred[0:1000,:])
import seaborn as sns
sns.heatmap(distmatrix)


from sklearn import metrics
labels_true = np.zeros(1000)

print(metrics.homogeneity_completeness_v_measure(labels_true,labels))
## How to get the number of points in certain category???