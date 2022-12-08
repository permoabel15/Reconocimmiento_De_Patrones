import numpy as np
import pandas as pd
from sklearn import metrics  # metricas
from sklearn.decomposition import PCA
from sklearn.feature_selection import SequentialFeatureSelector  # sel
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier  # knn
from sklearn.tree import DecisionTreeClassifier  # arbol

"Importacion de la base de datos desde un archivo .csv"
csv = pd.read_csv('csv/examen2Parcial.csv')
# print(csv)
features = ['A1', 'A2', 'A3', 'A4', 'A5', 'A6', 'A7', 'A8', 'A9']
target = ['Clase']
nFeatures = len(features)
csvP = pd.read_csv('csv/examen2Parcial.csv')

"Configuracion de algoritmo de k-Vecinos"
nn = 5
knn = KNeighborsClassifier(n_neighbors=nn)
knn.fit(csv[features], csv.iloc[:, nFeatures])
prediction = knn.predict(csvP[features])
confusion_matrix = metrics.confusion_matrix(csvP.iloc[:, nFeatures], prediction)
print("Matriz de confusion con ", nn, " Vecinos")
print(confusion_matrix)
print("Exactitud")
print(np.trace(confusion_matrix))
"Obtencion de las caracteristicas mas importantes por el metodo de seleccion hacia atras"
clas = KNeighborsClassifier(n_neighbors=nn)
sfs = SequentialFeatureSelector(clas, n_features_to_select=5, direction='backward')
sfs.fit(csv[features], csv.iloc[:, nFeatures])
print(sfs.get_support())
"Configuracion de algoritmo de k-Vecinos con las caracteristicas mas importantes"
kfeatures = ['A2', 'A3', 'A4', 'A5', 'A8']
nn = 5
knn = KNeighborsClassifier(n_neighbors=nn)
knn.fit(csv[kfeatures], csv.iloc[:, nFeatures])
prediction = knn.predict(csvP[kfeatures])
confusion_matrix = metrics.confusion_matrix(csvP.iloc[:, nFeatures], prediction)
print("")
print("")
print("Matriz de confusion con ", nn, " Vecinos y con las caracteristicas ",kfeatures)
print(confusion_matrix)
print("Exactitud")
print(np.trace(confusion_matrix))
print("******************************************************************************************************")
print("******************************************************************************************************")
"Configuracion de algoritmo de bayes"
gnb = GaussianNB()
gnb.fit(csv[features], csv.iloc[:, nFeatures])
prediction = gnb.predict(csvP[features])
confusion_matrix = metrics.confusion_matrix(csvP.iloc[:, nFeatures], prediction)
print('Matriz de confusion Bayes')
print(confusion_matrix)
print("Exactitud")
print(np.trace(confusion_matrix))
"Obtencion de las caracteristicas mas importantes por el metodo de seleccion hacia atras"
clas = GaussianNB()
sfs = SequentialFeatureSelector(clas, n_features_to_select=5, direction='backward')
sfs.fit(csv[features], csv.iloc[:, nFeatures])
print(sfs.get_support())
gfeatures = ['A2', 'A3', 'A5', 'A7', 'A9']

print("")
print("")
"Configuracion de algoritmo de bayes con las caracteristicas mas importantes"
gnb = GaussianNB()
gnb.fit(csv[gfeatures], csv.iloc[:, nFeatures])
prediction = gnb.predict(csvP[gfeatures])
confusion_matrix = metrics.confusion_matrix(csvP.iloc[:, nFeatures], prediction)
print('Matriz de confusion Bayes con las caracteristicas ',gfeatures)
print(confusion_matrix)
print("Exactitud")
print(np.trace(confusion_matrix))

print("******************************************************************************************************")
print("******************************************************************************************************")

"Configuracion de algoritmo del arbol"
dtree = DecisionTreeClassifier()
dtree = dtree.fit(csv[features], csv.iloc[:, nFeatures])
prediction = dtree.predict(csvP[features])
confusion_matrix = metrics.confusion_matrix(csvP.iloc[:, nFeatures], prediction)
print('Matriz de confusion Arbol')
print(confusion_matrix)
print("Exactitud")
print(np.trace(confusion_matrix))
"Obtencion de las caracteristicas mas importantes por el metodo de seleccion hacia atras"
clas = DecisionTreeClassifier()
sfs = SequentialFeatureSelector(clas, n_features_to_select=5, direction='backward')
sfs.fit(csv[features], csv.iloc[:, nFeatures])
print(sfs.get_support())
tfeatures = ['A2', 'A4', 'A5', 'A7', 'A9']
print("")
print("")
"Configuracion de algoritmo del arbol con las caracteristicas mas importantes"
dtree = DecisionTreeClassifier()
dtree = dtree.fit(csv[tfeatures], csv.iloc[:, nFeatures])
prediction = dtree.predict(csvP[tfeatures])
confusion_matrix = metrics.confusion_matrix(csvP.iloc[:, nFeatures], prediction)
print('Matriz de confusion Arbol con las caracteristicas ',tfeatures)
print(confusion_matrix)
print("Exactitud")
print(np.trace(confusion_matrix))
print("******************************************************************************************************")
print("******************************************************************************************************")
"Obtencion del porcentaje de explicacion de cada caracteristica"
x = csv[features]
y = csv[target]
pca = PCA(n_components=nFeatures)
X_r = pca.fit(x).transform(x)
print("Explicacion de cada componente: %s" % str(pca.explained_variance_ratio_))
