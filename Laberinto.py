# -*- coding: utf-8 -*-
"""
Created on Mon Apr  3 16:45:21 2023

@author: ESFM
"""

import random, datatime, csv, os 
from tkinter import * 
from enum import Enum
from collections import deque

#---------------------------------------------
#   Definición de paletas pares de colores 
#   Utiliza módulo enum
#---------------------------------------------

class COLOR(enum):
    dark = ('gray11','white')
    light = ('white','black')
    black = ('black','dim gray')
    red = ('red3','tomato')
    cyan = ('cyan4','cyan4')
    green = ('green4','pale green')
    blue = ('DeepskyBlue4','DeepSkyBlue2')
    yellow = ('yellow2','yellow2')
    
#-------------------------
# Agente de búsqueda 
#-------------------------
class agent:
    
    #-------------------------------+
    # Constructor del agente dentro del laberinto   parentMaze
    #-------------------------------+
    
    def __init__(self,parentMaze, x = None, y = None, shape = 'square', goal = None, filled = False, footprints  False
                 color: COLOR=COLOR.yellow):
        self._parentMaze=parentMaze
        self.color=color
        #----------------------------------
        # Checa que sean str los elementos de la tupla color
        #----------------------------------------------------
        if(isinstance(color,str)):
            if(color in COLOR.__members__):
                self.color= COLOR[color]
            else:
                raise ValueError(f'{color} is not a valid COLOR!')
        self.filled = filled
        self.shape = shape 
        self._open = 0
        if x is None:x = parentMaze.rows
        if y is None:y= parent
            
