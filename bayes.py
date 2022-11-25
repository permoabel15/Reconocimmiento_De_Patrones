import pandas as pd
import numpy
from scipy import stats
import matplotlib.pyplot as plt
from sklearn.preprocessing import KBinsDiscretizer
from sklearn.naive_bayes import GaussianNB #bayes
from sklearn import metrics #metricas

import csv

iris = pd.read_csv('csv/muestrasIris.csv')
print(iris)
features=['A1', 'A2', 'A3', 'A4']

gnb = GaussianNB()
gnb.fit(iris[features], iris.iloc[:,4])

irisP = pd.read_csv('csv/pruebaIris.csv')
prediction = gnb.predict(irisP[features])

confusion_matrix = metrics.confusion_matrix(irisP.iloc[:,4], prediction)
print(confusion_matrix)
