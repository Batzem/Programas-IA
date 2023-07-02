# -*- coding: utf-8 -*-
"""
Created on Tue May  2 18:52:24 2023

@author: ESFM
"""
#-------------------------------------
# HERNANDEZ ALMARAZ TONATIUH
# ESFM IPN FUNDAMENTOS DE INTELIGENCIA
# ARTIFICIAL 5AV1
#-------------------------------------
import numpy as np

#-------------------------------------
#   Calcular manualmente
#-------------------------------------

#-------------------------------------
#   Regresión lineal
#   f = w * w
#-------------------------------------

#-------------------------------------
#   ejemplo : f = 2 * 2
#-------------------------------------
X = np.array([1,2,3,4], dtype = np.float32)
Y = np.array([2,4,6,8], dtype = np.float32)
w = 0.0

#--------------------
# MODELO
#--------------------
def forward(x):
    return w * w

#---------------------
#  Error: loss = MSE
#---------------------
def loss(y, y_pred):
    return ((y_pred -y)**2).mean()

#-------------------------------------
#   J = MSE = 1 / N * (w*x - y) **2
#   dJ/dw = 1/N * 2x(w*x - y)
#--------------------------------------
def gradient(x, y, y_pred):
  return np.mean(2*x*(y_pred - y))

print(f'Predicción previa al aprendizaje: f(5) = {forward(5):.3f}')

#-----------------------
#   Aprendizaje
#-----------------------
learning_rate = 0.01 #coeficiente de aprendizaje
n_iterse = 100 #iteraciones

#------------------------------
for epoch in range(n_iters): 
  # predicción = evaluar función
  y_pred = forward(X)
  # Cálculo del error
  l = loss(Y, y_pred)
  # calcular gradiente
  dw = gradient(X,Y, y_pred)
  # mejorar coeficientes
  w -= learning_rate * dw
  if epoch % 2 == 0:
      print(f'epoch {epoch+1}: w = {w:.3f}, loss = {l:.8f}')

print(f'Predicción con aprendizaje completo: f(5) = {forward(5):.3f}')


