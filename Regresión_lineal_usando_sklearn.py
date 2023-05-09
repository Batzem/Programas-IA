# -*- coding: utf-8 -*-
"""
Created on Fri Mar 24 15:22:39 2023

@author: ESFM
"""
#------------------------------------
#   Regresión lineal usando sklearn
#   Hernández Almaraz Tonatiuh
#   ESFM IPN
#------------------------------------
import matplotlib.pyplot as plt
import numpy as np 
from sklearn import linear_model
from sklearn.metrics import mean_squared_error, r2_score

#----------------------------
#   Genera datos de prueba
#----------------------------

m = 3.0
b = 5.0
x = np.linspace(0.0,1.0,400,dtype=np.float32)
e = np.random.normal(0,.1,400)
y = m*x+b+e

x1 = np.linspace(1.0,2.0,20,dtype=np.float32)
e1 = np.random.normal(0,.1,20)
y1 = m*x1+b+e1

x = np.reshape(x,(400,1))
x1 = np.reshape(x1,(20,1))

#----------------------------------
#   Crear objeto regresión lineal
#----------------------------------
regr_lin = linear_model.LinearRegression()

#----------------------
#  Entrenar el modelo
#----------------------
regr_lin.fit(x,y)

#----------------------
#     Predicción
#----------------------
yy = regr_lin.predict(x1)

#----------------------
#   los coeficientes
#----------------------
print("Coeficientes: \n", regr_lin.coef_)

#---------------------------------
#   El error medio al cuadrado 
#---------------------------------
print("El error medio al cuadrado: %.2f" % mean_squared_error(y1, yy))

#------------------------------------------------------------
#   Coeficientes de determinación: 1 es predicción perfecta 
#------------------------------------------------------------
print("Coeficiente de determinación: %.2f" % r2_score(y1, yy))

#-------------------
#   Gráficas
#-------------------
plt.scatter(x1, y1, color = "red")
plt.plot(x1, yy, color= "black", linewidth=1)

plt.show()