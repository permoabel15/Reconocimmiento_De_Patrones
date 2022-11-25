from sklearn import tree  # arbol
from sklearn.tree import DecisionTreeClassifier  # arbol
import matplotlib.image as pltimg  # arbol
from sklearn.neighbors import KNeighborsClassifier  # knn
import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
from sklearn.preprocessing import KBinsDiscretizer
from sklearn import metrics  # metricas
import csv
from sklearn.decomposition import PCA

iris = pd.read_csv('csv/wdbcEntrenamiento.csv')
print(iris)
features = ['A1', 'A2', 'A3', 'A4', 'A5', 'A6', 'A7', 'A8', 'A9', 'A10', 'A11', 'A12', 'A13', 'A24', 'A15', 'A16',
            'A17', 'A18', 'A19', 'A20', 'A21', 'A22', 'A23', 'A24', 'A25', 'A26', 'A27', 'A28', 'A29', 'A30']
target = ['Clase']


X = iris[features]
y = iris[target]
target_names = ['setosa', 'virginica', 'versicolor']
"""
pca = PCA(n_components=4)
X_r = pca.fit(X).transform(X)


# Porcentaje de varianza explicado por cada componente
print(
    "Explicación de cada componente: %s"
    % str(pca.explained_variance_ratio_)
)

i1 = iris.loc[:, 'Clase'] == 1
i2 = iris.loc[i1]
i3 = iris.loc[:, 'Clase'] == 2
i4 = iris.loc[i3]
i5 = iris.loc[:, 'Clase'] == 3
i6 = iris.loc[i5]
plt.scatter(i2.iloc[:, 0], i2.iloc[:, 1], label='Iris-setosa')
plt.scatter(i4.iloc[:, 0], i4.iloc[:, 1], label='Iris-versicolor')
plt.scatter(i6.iloc[:, 0], i6.iloc[:, 1], label='Iris-virginica')
plt.title('Atributo A2 vs A3')
plt.xlabel('A2')
plt.ylabel('A3')
plt.legend()
plt.show()

i2 = X_r[i1]
i4 = X_r[i3]
i6 = X_r[i5]
plt.scatter(i2[:, 0], i2[:, 1], label='Iris-setosa')
plt.scatter(i4[:, 0], i4[:, 1], label='Iris-versicolor')
plt.scatter(i6[:, 0], i6[:, 1], label='Iris-virginica')
plt.title('Atributo Y1 vs Y2')
plt.xlabel('A2')
plt.ylabel('A3')
plt.legend()
plt.show()

for x in features:
    slope, intercept, r, p, std_err = stats.linregress(iris[x], iris['Clase'])
    print(r)
"""
dtree = DecisionTreeClassifier()
dtree = dtree.fit(iris[features], iris.iloc[:,30])

irisP = pd.read_csv('csv/wdbcPrueba.csv')
prediction = dtree.predict(irisP[features])
confusion_matrix = metrics.confusion_matrix(irisP.iloc[:,30], prediction)

print(confusion_matrix)

fig, axes = plt.subplots(nrows = 1,ncols = 1,figsize = (4,4), dpi=300)
tree.plot_tree(dtree, feature_names=features)
fig.savefig('image/wdbc.png')
