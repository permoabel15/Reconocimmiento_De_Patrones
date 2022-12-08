from sklearn.feature_selection import SequentialFeatureSelector  # sel
from sklearn import tree  # arbol
from sklearn.naive_bayes import GaussianNB
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

iris = pd.read_csv('csv/muestrasIris.csv')
# print(iris)
features = ['A1', 'A2', 'A3', 'A4']

nn = 3
clas = KNeighborsClassifier(n_neighbors=nn)
sfs = SequentialFeatureSelector(clas, n_features_to_select=3, direction='forward')
sfs.fit(iris[features], iris.iloc[:, 4])
print(sfs.get_support())

clas = DecisionTreeClassifier()
sfs = SequentialFeatureSelector(clas, n_features_to_select=3, direction='forward')
sfs.fit(iris[features], iris.iloc[:, 4])
print(sfs.get_support())

clas = GaussianNB()
sfs = SequentialFeatureSelector(clas, n_features_to_select=3, direction='forward')
sfs.fit(iris[features], iris.iloc[:, 4])
print(sfs.get_support())
