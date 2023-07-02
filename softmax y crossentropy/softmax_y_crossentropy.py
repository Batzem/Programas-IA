#-------------------------
# ESFM IPN FUNDAMENTOS IA
# HERNANDEZ TONATIUH 5AV1
#--------------------------

#-----------------------------------------------------------------
# Introducción al uso de softmax y crossentropy loss en pytorch
#-----------------------------------------------------------------
import torch
import torch.nn as nn
import numpy as np

#----------------------------------------------------------------------------
# Módelo de Boltzmann
#----------------------------
# En termodinámica es la probabilidad de encontrar un sistema
# ESTADO DADO SU ENERGÍA Y TEMPERATURA
#
#        -> 2.0              -> 0.65  
# Linear -> 1.0  -> Softmax  -> 0.25   -> CrossEntropy(y, y_hat)
#        -> 0.1              -> 0.1                   
#
#     puntajes(logits)      probabilidades
#                           sum = 1.0
#----------------------------------------------------------------------------

#------------------------------------------------------------------------------
# Softmax aplica el módelo de distribución exponencial para cada elemento
# normalizada con la suma de todas las exponenciales
#------------------------------------------------------------------------------
def softmax(x):
    return np.exp(x) / np.sum(np.exp(x), axis=0)

#----------------
# Vector en R3
#----------------
x = np.array([2.0, 1.0, 0.1])
#-----------------------------------
# softmax de elementos del vector
#-----------------------------------
outputs = softmax(x)
print('softmax numpy:', outputs)

x = torch.tensor([2.0, 1.0, 0.1])
outputs = torch.softmax(x, dim=0) # along values along first axis
print('softmax torch:', outputs)
#--------------------------------------
# Cross entropy
# Cross-entropy loss, or log loss, mide el rendimiento de un modelo de clasifcacion 
# cuya salida es un valor de probabilidad entre 0 y 1
#-------------------------------------------------------------------------------
# Se incrementa conforme la probabilidad diverge del nivel verdadero
#------------------------------------------------------------------------------- 

def cross_entropy(actual, predicted):
    EPS = 1e-15
    #limitar los valores a un mínimo EPS y máximo 1-EPS
    predicted = np.clip(predicted, EPS, 1 - EPS)
    # cálculod el rendimiento
    loss = -np.sum(actual * np.log(predicted))
    return loss # / float(predicted.shape[0])

#----------------------------------
# y debe ser alguna de las opciones
# if class 0: [1 0 0]
# if class 1: [0 1 0]
# if class 2: [0 0 1]
#------------------------------------
Y = np.array([1, 0, 0])
Y_pred_good = np.array([0.7, 0.2, 0.1])
Y_pred_bad = np.array([0.1, 0.3, 0.6])
l1 = cross_entropy(Y, Y_pred_good)
l2 = cross_entropy(Y, Y_pred_bad)
print(f'Loss1 numpy: {l1:.4f}')
print(f'Loss2 numpy: {l2:.4f}')

#--------------------------------------------------
# CrossEntropyLoss en PyTorch (aplica Softmax)
# nn.LogSoftmax + nn.NLLLoss
# NLLLoss = negative log likelihood loss
#--------------------------------------------------

loss = nn.CrossEntropyLoss()
# loss(input, target)

#------------------------------------------------------------------------
# el objetivo es de tamaño nSamples = 1
# cada elemento tiene etiqueta de clase: 0, 1, or 2
# Y (=objtivo) contiene etiquetas de clase  class no opciones binarias
#------------------------------------------------------------------------
Y = torch.tensor([0])

#-------------------------------------------------------------------------------------
# input es de tamaño nSamples x nClasses = 1 x 3
# y_pred (=input) deben estar sin normalizar (logits) para cada clase, no con softmax
#-------------------------------------------------------------------------------------
Y_pred_good = torch.tensor([[2.0, 1.0, 0.1]])
Y_pred_bad = torch.tensor([[0.5, 2.0, 0.3]])
# usar loss = nn.cossentropyloss()
l1 = loss(Y_pred_good, Y)
l2 = loss(Y_pred_bad, Y)

print(f'PyTorch Loss1: {l1.item():.4f}')
print(f'PyTorch Loss2: {l2.item():.4f}')

# predicciones (regresa el máximo de la dimensión)
_, predictions1 = torch.max(Y_pred_good, 1)
_, predictions2 = torch.max(Y_pred_bad, 1)
print(f'Actual class: {Y.item()}, Y_pred1: {predictions1.item()}, Y_pred2: {predictions2.item()}')

# permite calcular el rendimiento de múltiples conjuntos de datos

# vector objetivo es de tamaño nBatch = 3
# cada elemento tiene etiqueta de calse: 0, 1, or 2
Y = torch.tensor([2, 0, 1])

# Matriz input es de tamaño nBatch x nClasses = 3 x 3
# Y_pred son logits (no softmax)
Y_pred_good = torch.tensor(
    [[0.1, 0.2, 3.9], # predice clase 2
    [1.2, 0.1, 0.3], # predice clase 0
    [0.3, 2.2, 0.2]]) # predice clase 1

Y_pred_bad = torch.tensor(
    [[0.9, 0.2, 0.1],
    [0.1, 0.3, 1.5],
    [1.2, 0.2, 0.5]])

l1 = loss(Y_pred_good, Y)
l2 = loss(Y_pred_bad, Y)
print(f'Batch Loss1:  {l1.item():.4f}')
print(f'Batch Loss2: {l2.item():.4f}')
#----------------
#predicciones
#----------------
_, predictions1 = torch.max(Y_pred_good, 1)
_, predictions2 = torch.max(Y_pred_bad, 1)
print(f'Actual class: {Y}, Y_pred1: {predictions1}, Y_pred2: {predictions2}')

#-----------------------------------------
# clasificacion binaria de (red neuronal)
#-----------------------------------------
class NeuralNet1(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(NeuralNet1, self).__init__()
        self.linear1 = nn.Linear(input_size, hidden_size) 
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(hidden_size, 1)  
    
    def forward(self, x):
        out = self.linear1(x)
        out = self.relu(out)
        out = self.linear2(out)
        # sigmoide al final
        y_pred = torch.sigmoid(out)
        return y_pred

#-------------------------------------------
# correr problema de clasificación binaria
#-------------------------------------------
model = NeuralNet1(input_size=28*28, hidden_size=5)
criterion = nn.BCELoss()
#----------------------
# Multiples clases
#-----------------------
class NeuralNet2(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(NeuralNet2, self).__init__()
        self.linear1 = nn.Linear(input_size, hidden_size) 
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(hidden_size, num_classes)  
    
    def forward(self, x):
        out = self.linear1(x)
        out = self.relu(out)
        out = self.linear2(out)
        # sin softmax al final
        return out
#---------------------------------------
# correr problema de múltiples clases
#---------------------------------------
model = NeuralNet2(input_size=28*28, hidden_size=5, num_classes=3)
criterion = nn.CrossEntropyLoss()  # (aplica Softmax)

