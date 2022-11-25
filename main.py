import pandas as pd
import numpy
from scipy import stats
import matplotlib.pyplot as plt
from sklearn.preprocessing import KBinsDiscretizer
from sklearn.naive_bayes import GaussianNB  # bayes
from sklearn import metrics  # metricas
from scipy import stats
from sklearn.decomposition import PCA
import csv

iris = pd.read_csv('muestrasIris.csv')
features = ['A1', 'A2','A3', 'A4']
target = ['Clase']

X = iris[features]
y = iris[target]

# inicia configuración de PCA
target_names = ['setosa', 'virginica', 'versicolor']
pca = PCA(n_components=4)
X_r = pca.fit(X).transform(X)
# print(X_r)

# Porcentaje de varianza explicado por cada componente
print("Explicación de cada componente: %s"%str(pca.explained_variance_ratio_))

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

# print(iris)
