#--------------------------
# ESFM IPN FUNDAMENTOS IA
# TONATIUH HERNANDEZ
#--------------------------

import torch
import torch.nn as nn

#----------------------
# Regresión lineal
# f = w * x 
# f = 2 * x
#----------------------

#-------------------------------------------------------------
# 0) Datos de entrenamiento, cuidado con las dimensiones
#-------------------------------------------------------------
X = torch.tensor([[1], [2], [3], [4]], dtype=torch.float32)
Y = torch.tensor([[2], [4], [6], [8]], dtype=torch.float32)
n_samples, n_features = X.shape
print(f'#datos: {n_samples}, #características: {n_features}')

#-------------------------
# 0) Dato de prueba
#-------------------------
X_test = torch.tensor([5], dtype=torch.float32)

#------------------------------------------------------------
# 1) Diseño del modelo (debe contener la forma a evaluar)
# Podemos utilizar un módelo ya incluído en Pytorch
#------------------------------------------------------------

input_size = n_features
output_size = n_features

#-----------------------------------------------------------------
# Diseño del modelo (lineal) con dimensiones de entrada y salida
#-----------------------------------------------------------------

model = nn.Linear(input_size, output_size)

print(f'Predicción antes del aprendizaje: f(5) = {model(X_test).item():.3f}')

#--------------------------------
# 2) Definir error y optimizador
#---------------------------------
learning_rate = 0.01
n_iters = 100
# error
loss = nn.MSELoss()
#optimizador
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

#--------------------------
# 3) Ciclo de aprendizaje
#--------------------------
for epoch in range(n_iters):
    # predict = evaluar función (forward)
    y_predicted = model(X)
    # error
    l = loss(Y, y_predicted)
    # Calcular gradiente = Retropropagación (backward)
    l.backward()
    # mejorar coeficientes
    optimizer.step()
    # reseetear coeficientes
    optimizer.zero_grad()
    #diagnóstico
    if epoch % 10 == 0:
        [w, b] = model.parameters() # obtener parámetros
        print('epoch ', epoch+1, ': w = ', w[0][0].item(), ' loss = ', l)

print(f'Predicción después del aprendizaje: f(5) = {model(X_test).item():.3f}')
