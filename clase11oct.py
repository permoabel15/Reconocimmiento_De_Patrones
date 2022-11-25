from cProfile import label
import pandas as pd
import numpy
from scipy import stats
import matplotlib.pyplot as plt

#importar archivo CSV
iris = pd.read_csv('csv/iris.csv')
#print(iris)

plt.hist(iris.iloc[:,4])
plt.show()

print(numpy.mean(iris.iloc[:,0:4]))
print("\n")
print(numpy.median(iris.iloc[:,0]))
print(numpy.median(iris.iloc[:,1]))
print(numpy.median(iris.iloc[:,2]))
print(numpy.median(iris.iloc[:,3]))
print("\n")
print(numpy.std(iris.iloc[:,0:4]))
print("\n")
print(min(iris.iloc[:,0]))
print(max(iris.iloc[:,0]))
print("\n")
print(min(iris.iloc[:,1]))
print(max(iris.iloc[:,1]))
print("\n")
print(min(iris.iloc[:,2]))
print(max(iris.iloc[:,2]))
print("\n")
print(min(iris.iloc[:,3]))
print(max(iris.iloc[:,3]))


print("\n")
i1=iris.loc[:,'Clase']=='Iris-setosa'
i2=iris.loc[i1]
print(i2)

print("\n")
i3=iris.loc[:,'Clase']=='Iris-versicolor'
i4=iris.loc[i3]
print(i4)

print("\n")
i5=iris.loc[:,'Clase']=='Iris-virginica'
i6=iris.loc[i5]
print(i6)

plt.title("A1,A2")
plt.scatter(i2.iloc[:,1],i2.iloc[:,2],label="Setosa")
plt.scatter(i4.iloc[:,1],i4.iloc[:,2],label="Versicolor")
plt.scatter(i6.iloc[:,1],i6.iloc[:,2],label="Virginica")
plt.xlabel("Ancho Sepalo")
plt.ylabel("Longitud Petalo")
plt.legend()
plt.show()

plt.title("A1,A3")
plt.scatter(i2.iloc[:,1],i2.iloc[:,3],label="Setosa")
plt.scatter(i4.iloc[:,1],i4.iloc[:,3],label="Versicolor")
plt.scatter(i6.iloc[:,1],i6.iloc[:,3],label="Virginica")
plt.legend()
plt.show()

plt.title("A2,A3")
plt.scatter(i2.iloc[:,2],i2.iloc[:,3],label="Setosa")
plt.scatter(i4.iloc[:,2],i4.iloc[:,3],label="Versicolor")
plt.scatter(i6.iloc[:,2],i6.iloc[:,3],label="Virginica")
plt.legend()
plt.show()

plt.waitKey(0)
plt.destroyAllWindows()
