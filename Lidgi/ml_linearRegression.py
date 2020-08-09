# -*- coding: utf-8 -*-
"""
Created on Sat Aug  8 13:21:51 2020

@author: acer
"""

#Regresión lineal simple

import numpy as np
from sklearn import datasets, linear_model
import matplotlib.pyplot as plt

boston = datasets.load_boston()
print(boston.keys())
print()

print(boston.DESCR)
print(boston.data.shape)
print(boston.feature_names)

X = boston.data[:, np.newaxis, 5]
y = boston.target

plt.scatter(X,y)
plt.xlabel('Número de habitaciones')
plt.ylabel('Valor medio')
plt.show()

#Implementacion del modelo de regresión lineal simple
from sklearn.model_selection import train_test_split

#Separar datos de entrenamiento y de prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
#Definir el modelo
lr = linear_model.LinearRegression()
#Entrenar modelo
lr.fit(X_train, y_train)
#Predicción
Y_pred = lr.predict(X_test)

print(lr.coef_)
print(lr.intercept_)
print(lr.score(X_train,y_train))

#Regresión lineal múltiple
#Seleccionamos las columna 5, 6 y 7 del dataset
X_multiple = boston.data[:, 5:8]

#Defino los datos correspondientes a las etiquetas
y_multiple = boston.target

#Separo los datos de "train" en entrenamiento y prueba para probar los algoritmos
X_train, X_test, y_train, y_test = train_test_split(X_multiple, y_multiple, test_size=0.2)

#Defino el algoritmo a utilizar
lr_multiple = linear_model.LinearRegression()

#Entreno el modelo
lr_multiple.fit(X_train, y_train)

#Realizo una predicción
Y_pred_multiple = lr_multiple.predict(X_test)

print('DATOS DEL MODELO REGRESIÓN LINEAL MULTIPLE')
print()
print('Valor de las pendientes o coeficientes "a":')
print(lr_multiple.coef_)
print('Valor de la intersección o coeficiente "b":')
print(lr_multiple.intercept_)

print('Precisión del modelo:')
print(lr_multiple.score(X_train, y_train))


    




