from ast import iter_fields
from cProfile import label
import pandas as pd
import numpy
from scipy import stats
import matplotlib.pyplot as plt

#importar archivo CSV
wineRed = pd.read_csv('csv/winequality-red.csv')
wineWhite = pd.read_csv('csv/winequality-white.csv')



#plt.title("Histograma Calidad de Vinos tintos")
#plt.hist(wineRed.iloc[:,11])
#plt.show()

#plt.title("Histograma Calidad de Vinos blancos")
#plt.hist(wineWhite.iloc[:,11])
#plt.show()

print("Media de la base de datos de los vinos tintos")
print(wineRed.mean())
print("\n")
print("Desviacion Estandar de la base de datos de los vinos tinto")
print(wineRed.std())
print("\n")
print("Mediana de los atributos de los vinos tintos")
print(wineRed.median())
print("Minimo de los atributos de los vinos tintos")
print(wineRed.min())
print("Maximo de los atributos de los vinos tintos")
print(wineRed.max())
print("\n")



print("Media de la base de datos de los vinos blancos")
print(wineWhite.mean())
print("\n")
print("Desviacion Estandar de la calidad de los vinos Blancos")
print(wineWhite.std())
print("\n")
print("Mediana de los atributos de los vinos blancos")
print(wineWhite.median())
print("\n")
print("Minimo de los atributos de los vinos blancos")
print(wineWhite.min())
print("\n")
print("Maximo de los atributos de los vinos blancos")
print(wineWhite.max())
print("\n")

wineRQ1=wineRed.loc[:,'quality']==1
wineRQ2=wineRed.loc[:,'quality']==2
wineRQ3=wineRed.loc[:,'quality']==3
wineRQ4=wineRed.loc[:,'quality']==4
wineRQ5=wineRed.loc[:,'quality']==5
wineRQ6=wineRed.loc[:,'quality']==6
wineRQ7=wineRed.loc[:,'quality']==7
wineRQ8=wineRed.loc[:,'quality']==8
wineRQ9=wineRed.loc[:,'quality']==9

wineWQ1=wineWhite.loc[:,'quality']==1
wineWQ2=wineWhite.loc[:,'quality']==2
wineWQ3=wineWhite.loc[:,'quality']==3
wineWQ4=wineWhite.loc[:,'quality']==4
wineWQ5=wineWhite.loc[:,'quality']==5
wineWQ6=wineWhite.loc[:,'quality']==6
wineWQ7=wineWhite.loc[:,'quality']==7
wineWQ8=wineWhite.loc[:,'quality']==8
wineWQ9=wineWhite.loc[:,'quality']==9

for i in range(5):
    clase1=wineRed.columns[i]
    clase2=wineRed.columns[i+1]
    plt.title('Vino Tinto A '+ clase1+' B '+clase2)
    plt.scatter(wineRed.loc[wineRQ1].iloc[:,i],wineRed.loc[wineRQ1].iloc[:,i+1],label="Calidad 1")
    plt.scatter(wineRed.loc[wineRQ2].iloc[:,i],wineRed.loc[wineRQ2].iloc[:,i+1],label="Calidad 2")
    plt.scatter(wineRed.loc[wineRQ3].iloc[:,i],wineRed.loc[wineRQ3].iloc[:,i+1],label="Calidad 3")
    plt.scatter(wineRed.loc[wineRQ4].iloc[:,i],wineRed.loc[wineRQ4].iloc[:,i+1],label="Calidad 4")
    plt.scatter(wineRed.loc[wineRQ5].iloc[:,i],wineRed.loc[wineRQ5].iloc[:,i+1],label="Calidad 5")
    plt.scatter(wineRed.loc[wineRQ6].iloc[:,i],wineRed.loc[wineRQ6].iloc[:,i+1],label="Calidad 6")
    plt.scatter(wineRed.loc[wineRQ7].iloc[:,i],wineRed.loc[wineRQ7].iloc[:,i+1],label="Calidad 7")
    plt.scatter(wineRed.loc[wineRQ8].iloc[:,i],wineRed.loc[wineRQ8].iloc[:,i+1],label="Calidad 8")
    plt.scatter(wineRed.loc[wineRQ9].iloc[:,i],wineRed.loc[wineRQ9].iloc[:,i+1],label="Calidad 9")
    plt.xlabel(clase1)
    plt.ylabel(clase2)
    plt.legend()
    plt.show()

for i in range(5):
    clase1=wineWhite.columns[i]
    clase2=wineWhite.columns[i+1]
    plt.title('Vino Blanco A '+ clase1+' B '+clase2)
    plt.scatter(wineWhite.loc[wineWQ1].iloc[:,i],wineWhite.loc[wineWQ1].iloc[:,i+1],label="Calidad 1")
    plt.scatter(wineWhite.loc[wineWQ2].iloc[:,i],wineWhite.loc[wineWQ2].iloc[:,i+1],label="Calidad 2")
    plt.scatter(wineWhite.loc[wineWQ3].iloc[:,i],wineWhite.loc[wineWQ3].iloc[:,i+1],label="Calidad 3")
    plt.scatter(wineWhite.loc[wineWQ4].iloc[:,i],wineWhite.loc[wineWQ4].iloc[:,i+1],label="Calidad 4")
    plt.scatter(wineWhite.loc[wineWQ5].iloc[:,i],wineWhite.loc[wineWQ5].iloc[:,i+1],label="Calidad 5")
    plt.scatter(wineWhite.loc[wineWQ6].iloc[:,i],wineWhite.loc[wineWQ6].iloc[:,i+1],label="Calidad 6")
    plt.scatter(wineWhite.loc[wineWQ7].iloc[:,i],wineWhite.loc[wineWQ7].iloc[:,i+1],label="Calidad 7")
    plt.scatter(wineWhite.loc[wineWQ8].iloc[:,i],wineWhite.loc[wineWQ8].iloc[:,i+1],label="Calidad 8")
    plt.scatter(wineWhite.loc[wineWQ9].iloc[:,i],wineWhite.loc[wineWQ9].iloc[:,i+1],label="Calidad 9")
    plt.xlabel(clase1)
    plt.ylabel(clase2)
    plt.legend()
    plt.show()