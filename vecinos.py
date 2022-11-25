from sklearn.neighbors import KNeighborsClassifier  # knn
import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
from sklearn.preprocessing import KBinsDiscretizer
from sklearn import metrics  # metricas
import csv

iris = pd.read_csv('csv/muestrasIris.csv')
print(iris)
features = ['A1', 'A2', 'A3', 'A4']


knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(iris[features], iris.iloc[:, 4])
irisP = pd.read_csv('csv/pruebaIris.csv')
prediction = knn.predict(irisP[features])
confusion_matrix = metrics.confusion_matrix(irisP.iloc[:, 4], prediction)

print(confusion_matrix)
