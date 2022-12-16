from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import csv
import pandas as pd
from sklearn import metrics #metricas

iris = pd.read_csv('csv/wdbc-ClasesMod.csv')
features=['A1','A2','A3', 'A4','A5', 'A6', 'A7', 'A8', 'A9','A10', 'A11', 'A12', 'A13', 'A14', 'A15', 'A16', 'A17', 'A18', 'A19', 'A20', 'A21', 'A22', 'A23', 'A24', 'A25', 'A26', 'A27', 'A28', 'A29','A30']

print(iris)
#crear 3 ccluster
kmeans = KMeans(n_clusters=2)
kmeans.fit(iris[features])

print(kmeans.labels_) #en que cluster clasifico casfa muestra 

confusion_matrix = metrics.confusion_matrix(iris['Clase'], kmeans.labels_)
print(confusion_matrix)


plt.subplot(1, 2, 1)
plt.scatter(iris['A1'], iris['A2'], c=kmeans.labels_)
plt.subplot(1, 2, 2)
plt.scatter(iris['A1'], iris['A2'], c=iris['Clase'])
plt.show()