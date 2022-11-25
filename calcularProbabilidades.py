from ast import iter_fields
from cProfile import label
import pandas as pd
import numpy
from scipy import stats
import matplotlib.pyplot as plt
computadora=pd.read_csv("csv/bayes.csv")
total_columns=len(computadora.axes[1])
total= len(computadora.axes[0])
##print(len(computadora.axes[1]))
columns =[]
conteo=[]
for i in range(total_columns):
    columns.append(computadora.iloc[:,i])
    conteo.append(columns[i].value_counts())

num=conteo[total_columns-1]
total_clases=len(conteo[total_columns-1])
print(total_clases)
p_Clases=[]

for i in range(total_clases):
    p_Clases.append(num[i]/total)

print(p_Clases)
