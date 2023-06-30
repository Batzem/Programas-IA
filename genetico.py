# -*- coding: utf-8 -*-
"""
Created on Tue Jun 13 17:58:57 2023

@author: ESFM
"""
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

    