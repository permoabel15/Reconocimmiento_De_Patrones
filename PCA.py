import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import csv

iris = pd.read_csv('csv/wdbcEntrenamiento.csv')
#print(iris)
features = ['A1', 'A2', 'A3', 'A4', 'A5', 'A6', 'A7', 'A8', 'A9', 'A10', 'A11', 'A12', 'A13', 'A24', 'A15', 'A16',
            'A17', 'A18', 'A19', 'A20', 'A21', 'A22', 'A23', 'A24', 'A25', 'A26', 'A27', 'A28', 'A29', 'A30']
target = ['Clase']


X = iris[features]
y = iris[target]
target_names = ['Maligno', 'Benigno']

pca = PCA(n_components=30)
X_r = pca.fit(X).transform(X)



# Porcentaje de varianza explicado por cada componente
#print(
#    "Explicación de cada componente: %s"
#    % str(pca.explained_variance_ratio_)
#)
for i in range(len(pca.explained_variance_ratio_)):
    print("{:.20f}".format(pca.explained_variance_ratio_[i]))


"""
with open('wdbcEntrenamientoPCA.csv', 'w', newline='', encoding='utf-8') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerows(X_r)
    

i1 = iris.loc[:, 'Clase'] == 'M'
i2 = iris.loc[i1]
i3 = iris.loc[:, 'Clase'] == 'B'
i4 = iris.loc[i3]
plt.scatter(i2.iloc[:, 0], i2.iloc[:, 1], label='Maligno')
plt.scatter(i4.iloc[:, 0], i4.iloc[:, 1], label='Benigno')
plt.title('Atributo A2 vs A3')
plt.xlabel('A2')
plt.ylabel('A3')
plt.legend()
plt.show()

i2 = X_r[i1]
i4 = X_r[i3]
plt.scatter(i2[:, 0], i2[:, 1], label='Maligno')
plt.scatter(i4[:, 0], i4[:, 1], label='Benigno')
plt.title('Atributo Y1 vs Y2')
plt.xlabel('A2')
plt.ylabel('A3')
plt.legend()
plt.show()

for x in features:
    slope, intercept, r, p, std_err = stats.linregress(iris[x], iris['Clase'])
    print(r)

"""
