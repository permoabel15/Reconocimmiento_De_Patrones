import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn import tree  #arbol
from sklearn.tree import DecisionTreeClassifier #arbol
from sklearn import metrics #metricas


iris = pd.read_csv('csv/wdbcEntrenamiento.csv')
print(iris)
features = ['A1', 'A2', 'A3', 'A4', 'A5', 'A6', 'A7', 'A8', 'A9', 'A10', 'A11', 'A12', 'A13', 'A24', 'A15', 'A16',
            'A17', 'A18', 'A19', 'A20', 'A21', 'A22', 'A23', 'A24', 'A25', 'A26', 'A27', 'A28', 'A29', 'A30']
#features=['A4']
featuresSel=['A1']

def seleccionCaracteristicas(features):
    x = 1
    acc = 0
    accN = 0
    accMax = []

    while True:
        acc = accN
        dtree = DecisionTreeClassifier()
        dtree = dtree.fit(iris[features[0:x]], iris.iloc[:, 30])
        irisP = pd.read_csv('csv/wdbcPrueba.csv')
        prediction = dtree.predict(irisP[features[0:x]])
        confusion_matrix = metrics.confusion_matrix(irisP.iloc[:, 30], prediction)
        print(confusion_matrix)
        accN = (np.trace(confusion_matrix)) / 120
        print(accN)
        if accN <= acc or x >= len(features):
            x -= 1
            break

        x += 1
    print(features[0:x])
    fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(4, 4), dpi=300)
    tree.plot_tree(dtree, feature_names=features)