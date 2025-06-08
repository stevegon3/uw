import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.cluster import AgglomerativeClustering
from sklearn.preprocessing import StandardScaler
import scipy.cluster.hierarchy as sch

data = pd.read_excel('user_knowledge.xlsx')

X = data[['STG','SCG','STR','LPR','PEG']]
scaler = StandardScaler()
scaler.fit_transform(X)

kmeans = KMeans(n_clusters=4)
y_means = kmeans.fit_predict(X)

centroids = kmeans.cluster_centers_
print(centroids)
print(y_means)

# Visualize the clusters
plt.figure(figsize=(10,10))
plt.title('Divisive clustering with k-means')
plt.scatter(X['STG'], X['STR'], c=y_means, cmap='rainbow')
plt.scatter(centroids[:,0], centroids[:,2], c='black',s=100)
plt.show()

plt.figure(figsize=(10,10))
plt.title('Agglomerative clustering')
Dendrogram = sch.dendrogram((sch.linkage(X, method='ward')))

ac = AgglomerativeClustering(n_clusters=4)
y_ac = ac.fit_predict(X)

print(y_ac)
