from sklearn.neighbors import KNeighborsClassifier  # knn
import pandas as pd
from sklearn import metrics

iris = pd.read_csv('csv/wdbcEntrenamiento.csv')
print(iris)
features = ['A1', 'A2', 'A3', 'A4', 'A5', 'A6', 'A7', 'A8', 'A9', 'A10', 'A11', 'A12', 'A13', 'A24', 'A15', 'A16',
            'A17', 'A18', 'A19', 'A20', 'A21', 'A22', 'A23', 'A24', 'A25', 'A26', 'A27', 'A28', 'A29', 'A30', ]
nn=30
knn = KNeighborsClassifier(n_neighbors=nn)
knn.fit(iris[features], iris.iloc[:, 30])
irisP = pd.read_csv('csv/wdbcPrueba.csv')
prediction = knn.predict(irisP[features])
confusion_matrix = metrics.confusion_matrix(irisP.iloc[:, 30], prediction)

print("Numero de vecinos:",nn)
print(confusion_matrix)
