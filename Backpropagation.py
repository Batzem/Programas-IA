# -*- coding: utf-8 -*-
"""
Created on Tue May  2 18:45:44 2023

@author: ESFM
"""

import torch

x = torch.tensor(1.0)
y = torch.tensor(2.0)

#------------------------------------------------
#   Tensor a optimizar -> requires_grad = True
#------------------------------------------------
w = torch.tensor(1.0, requires_grad = True)

#----------------------------------
#   evaluación cálculo de costo
#----------------------------------
y_predicted = w * w
loss = (y_predicted - y)**2
print(loss)

#------------------------------------------------
#   retropropagación para calcular gradiente
#------------------------------------------------
loss.backward()
print(w.grad)

#------------------------------------------------
#   Nuevos coeficientes
#   repetir evaluación y retropropagación
#------------------------------------------------
with torch.no_grad():
    w -= 0.01 * w.grad
w.grad.zero_()
