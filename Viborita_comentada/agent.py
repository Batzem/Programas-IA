#----------------------------------#
#   AGENTE: VIBORITA INTELIGENTE   #
#----------------------------------#
#   Traductor: Tonatiuh Hernández  #
#     Inteligencia Artificial      #
#          ESFM IPN 2023           #
#----------------------------------#

#----------------------------------#
#         LIBRERIAS                #
#----------------------------------#

#Para poder ejecutar la inteligencia artificial
import torch
#Para generar números aleatorios"
import random
#Para trabajar con arreglos"
import numpy as np
#Para trabajar con pilas"
from collections import deque
from game import SnakeGameAI, Direction, Point
from model import Linear_QNet, QTrainer
from helper import plot

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

MAX_MEMORY = 100_000
BATCH_SIZE = 1000
LR = 0.001


#--------------------------------#
#      CLASE AGENTE              #
#--------------------------------#
class Agent:

    #----------------------------#
    #         Constructor        #
    #   modelo: red neuronal     #
    #   Entrenador - Optimizador #
    #----------------------------#
    def __init__(self):
        
        self.n_games = 0
        self.epsilon = 0 # Juegos al azar
        self.gamma = 0.9 # Tasa de descuento
        self.memory = deque(maxlen=MAX_MEMORY) #Pila finita popletf()
        self.model = Linear_QNet(11, 256, 3)
        self.trainer = QTrainer(self.model, lr=LR, gamma=self.gamma)

    #Definir método: obtener agente(2 argumentos, si mismo y el juego)"
    

    #--------------------------------------------------#
    #            MÉTODOS DE LA CLASE AGENTE            #
    #--------------------------------------------------#
    def get_state(self, game):
        
        head = game.snake[0]
        
        #-----------------#
        #   Pixel 20x20   #
        #-----------------#
        
        point_l = Point(head.x - 20, head.y)
        point_r = Point(head.x + 20, head.y)
        point_u = Point(head.x, head.y - 20)
        point_d = Point(head.x, head.y + 20)
        
        dir_l = game.direction == Direction.LEFT
        dir_r = game.direction == Direction.RIGHT
        dir_u = game.direction == Direction.UP
        dir_d = game.direction == Direction.DOWN

        state = [
            #Peligro enfrente
            (dir_r and game.is_collision(point_r)) or 
            (dir_l and game.is_collision(point_l)) or 
            (dir_u and game.is_collision(point_u)) or 
            (dir_d and game.is_collision(point_d)),
            
            #Peligro a la derecha
            (dir_u and game.is_collision(point_r)) or 
            (dir_d and game.is_collision(point_l)) or 
            (dir_l and game.is_collision(point_u)) or 
            (dir_r and game.is_collision(point_d)),
            
            #Peligro a la izquierda"
            (dir_d and game.is_collision(point_r)) or 
            (dir_u and game.is_collision(point_l)) or 
            (dir_r and game.is_collision(point_u)) or 
            (dir_l and game.is_collision(point_d)),
            
        
            # direcciones de movimiento
            dir_l,
            dir_r,
            dir_u,
            dir_d,
            
            #Localización de la comida
            game.food.x < game.head.x,  # comida a la izquierda
            game.food.x > game.head.x,  # comida a la derecha
            game.food.y < game.head.y,  # comida arriba
            game.food.y > game.head.y  # comida abajo
            ]

        return np.array(state, dtype=int)


    
    
    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done)) # popleft if MAX_MEMORY is reached
    
    #-----------------------------------------"
    #     Entrenar memoria de largo plazo     "
    #-----------------------------------------"
    def train_long_memory(self):
        if len(self.memory) > BATCH_SIZE:
            mini_sample = random.sample(self.memory, BATCH_SIZE) # lista de tuplas
        else:
            mini_sample = self.memory

        states, actions, rewards, next_states, dones = zip(*mini_sample)
        self.trainer.train_step(states, actions, rewards, next_states, dones)
        #for state, action, reward, nexrt_state, done in mini_sample:
        #    self.trainer.train_step(state, action, reward, next_state, done)
    
    #------------------------------------------"
    #     Entrenar memoria de corto plazo      "
    #------------------------------------------"
    
    def train_short_memory(self, state, action, reward, next_state, done):
        self.trainer.train_step(state, action, reward, next_state, done)

    #------------------------------------------"
    #           Decidir accion                 "
    #------------------------------------------"
    
    
    def get_action(self, state):
        
        #movimientos al azar"
        # random moves: tradeoff exploration / exploitation
        
        self.epsilon = 80 - self.n_games
        final_move = [0,0,0]
        if random.randint(0, 200) < self.epsilon:
        #---------------------------------------------------#    
        #       Genera entero al azar entre 0 y 2           #
        #---------------------------------------------------#
            move = random.randint(0, 2)
            final_move[move] = 1
        else:
        #---------------------------------------------------#    
        #     Dado state(R11) genera predicción (R3)        #
        #---------------------------------------------------#
            state0 = torch.tensor(state, dtype=torch.float)
            prediction = self.model(state0)
        #---------------------------------------------------#
        #         Move es entero entre 0 y 2                #
        # es la entrada de valor mpáximo en predicción (R3) #
        #---------------------------------------------------#
            move = torch.argmax(prediction).item()
            final_move[move] = 1
        #---------------------------------------------------#
        #     Desicion es un vector en R3 de con 0 o        #
        #---------------------------------------------------#
            
            
            
        return final_move

#------------------------------------------------"
#        FUNCIÓN PRINCIPAL: ENTRENAMIENTO        "
#------------------------------------------------"
def train():
    #lista de records
    plot_scores = []
    plot_mean_scores = []
    total_score = 0
    record = 0
    #Guarda en una variable el método agente"
    agent = Agent()
    #Guarda en una variable el metodo SnakeGameAI()"
    game = SnakeGameAI()
    while True:
        #Obtener estado pasado"
        state_old = agent.get_state(game)

        #Moverse
        final_move = agent.get_action(state_old)
        
        #Realizar un movimiento y obtener el nuevo estado.
        reward, done, score = game.play_step(final_move)
        state_new = agent.get_state(game)

        #entrenar memoria de corto plazo 
        agent.train_short_memory(state_old, final_move, reward, state_new, done)
        #Recordar
        agent.remember(state_old, final_move, reward, state_new, done)

        if done:
            
            #Entremar memoria de largo plazo, graficar resultado.
            game.reset()
            agent.n_games += 1
            agent.train_long_memory()

            if score > record:
                record = score
                agent.model.save()

            print('Game', agent.n_games, 'Score', score, 'Record:', record)

            plot_scores.append(score)
            total_score += score
            mean_score = total_score / agent.n_games
            plot_mean_scores.append(mean_score)
            plot(plot_scores, plot_mean_scores)


if __name__ == '__main__':
    train()
