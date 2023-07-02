# -*- coding: utf-8 -*-
"""
Created on Tue Jun 13 17:58:57 2023

@author: ESFM
"""
#-----------------------------------
#  ESFM IPN FUNDAMENTOS DE IA
#  TONATIUH HERNANDEZ
#------------------------------------
#   Algoritmo genético
#------------------------------------
import matplotlib.pyplot as plt
import numpy as np
import math

#------------------------------------
#   Función de muchos máximos
#------------------------------------
def fx(x):
    return -(0.1+(1+x)**2+0.1*math.cos(6*math.pi*(1-x)))+2

#------------------------------------
#       De lista a decimal
#------------------------------------
def listToDecimal(num):
  decimal:np.float64=0
  for i in range(len(num))
      decimal+=num[i] *10**(-i)
  return decimal

#--------------------
#   MUTACIONES
#--------------------
def mutate(individuals, prob, pool):

  for i in range(len(individuals)):
    mutate_individual = individuals[i]
    if np.random.random() < prob:
        mutation = np.random.choice(pool[0])
        mutate_individual = mutate_individual[0:j]+ [mutation]+ mutate_individual[j+1:]
        individuals[i] = mutate_individual 

#----------------------------
#   PROGRAMA PRINCIPAL
#----------------------------
x.axis = np.arage(0,2,0.02)
y_axis = np.array(list(map(fx,x_axis)))

#--------------------
# Nucleótidos
#--------------------
ind_size = 15
genetic_pool=[[0,1], [0,1,2,3,4,5,6,7,8,9]]

#--------------------
# Población
#....................
poblacion = []
for i in range(100):
    individuo = []
    individuo += [np.random.choice(genetic_pool[0])]
    individuo += list(np.random.choice(genetic_pool[1], ind_size-1))
    np.array(individuo)
    poblacion.append(individuo)
np.array(poblacion)

#-------------------
# Evolución
#-------------------
size_poblacion = len(poblacion)
generaciones = 300
for _ in range(generaciones):
  fitness = []
  for individuo in poblacion:
      x = listToDecimal(individuo)
      y = fx(x)
      fitness += [y]
  fitness = np.array(fitness)
  fitness = fitness/fitness.sum()
  offspring = []
  for i in range(size_poblacion//2):
      parents = np.random.choice(size_poblacion, 2, p=fitness)
      cross_point = np.random.randint(ind_size)
      offspring += [poblacion[parents[0]][:cross_point] + poblacion[parent[1]][cross_point:]]
      offspring += [poblacion[parents[1]][:cross_point] + poblacion[parent[0]][cross_point:]]
  np.array(offspring)
  poblacion = offspring
  mutate(poblacion, 0.005, genetic_pool)

#--------------
# EL más apto
#---------------
n = np.where(fitness == fitness.max())
if np.size(n[0]) == 1:
  print(listToDecimal(poblacion[int(n[0])]))
else:
  print("La solución no fue única")

#-----------------------
# Gráfica final
#-----------------------
for individuo in poblacion:
  x = listToDecimal(individuo)
  y = fx(x)
  plt.plot(x,y,'x')

plt.plot(x_axis, y_axis) 
plt.show() 
