import pandas as pd
import numpy
from scipy import stats
import matplotlib.pyplot as plt
from sklearn.preprocessing import KBinsDiscretizer
from sklearn.naive_bayes import GaussianNB #bayes
from sklearn import metrics #metricas

import csv

iris = pd.read_csv('csv/wdbcEntrenamiento.csv')
print(iris)
features = ['A1', 'A2', 'A3', 'A4', 'A5', 'A6', 'A7', 'A8', 'A9', 'A10', 'A11', 'A12', 'A13', 'A24', 'A15', 'A16',
            'A17', 'A18', 'A19', 'A20', 'A21', 'A22', 'A23', 'A24', 'A25', 'A26', 'A27', 'A28', 'A29', 'A30']

gnb = GaussianNB()
gnb.fit(iris[features], iris.iloc[:,30])

irisP = pd.read_csv('csv/wdbcPrueba.csv')
prediction = gnb.predict(irisP[features])

confusion_matrix = metrics.confusion_matrix(irisP.iloc[:,30], prediction)
print(confusion_matrix)
