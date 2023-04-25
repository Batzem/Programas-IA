# -*- coding: utf-8 -*-
"""
Created on Wed Mar 22 17:09:10 2023

@author: ESFM
"""
#---------------------------------------
# Regresión lineal
# Tonatiuh Quetzalli Hernández Almaraz
# Becerra Sagredo Julian Tercero
# Fundamentos de inteligencia Artificial
# ESFM IPN
#---------------------------------------
import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error, r2_score

#---------------------------------------
# Carga de datos de prueba (diabetes)
#---------------------------------------
diabetes_X, diabetes_y = datasets.load_diabetes(return_X_y=True)

#---------------------------------------
# Utilizar solo una parte de los datos
#---------------------------------------
diabetes_X = diabetes_X[:, np.newaxis, 2]

#-----------------------------------------------------
#Separar datos en conjuntos  de entrenamiento/prueba
#-----------------------------------------------------
diabetes_X_train = diabetes_X[:-20]
diabetes_X_test = diabetes_X[-20:]

#-----------------------------------------------------------
#Separar resultados en  conjuntos de entrenamiento / prueba
#-----------------------------------------------------------
diabetes_y_train = diabetes_y[:-20]
diabetes_y_test = diabetes_y[-20:]

#---------------------------------
# Crear objeto regresión lineal
#---------------------------------
regr_lin = linear_model.LinearRegression()

#------------------------------
# Entrenar el modelo
#------------------------------
regr_lin.fit(diabetes_X_train, diabetes_y_train)

#------------------------------
# Predicción
#------------------------------
diabetes_y_pred = regr_lin.predict(diabetes_X_test)

#------------------------------
# Los coeficientes
#------------------------------
print("Coeficientes: \n", regr_lin.coef_)
#----------------------------------------
# Error medio al cuadrado
#----------------------------------------
print("Error medio al cuadrado: %.2f" % mean_squared_error(diabetes_y_test, diabetes_y_pred))
#-----------------------------------------
# Coeficientes de determinación: 1 es predicción perfecta 
#------------------------------------------------------------
print(" Coeficiente de determinación: %.2f" %  r2_score(diabetes_y_test, diabetes_y_pred))

#-----------------------
# Gráficas
#-----------------------
plt.scatter(diabetes_X_test, diabetes_y_test, color = "black")
plt.plot(diabetes_X_test, diabetes_y_pred, color = "blue", linewidth=3)

plt.xticks(())
plt.yticks(())

plt.show()













