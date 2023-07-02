import pygame
import random
from enum import Enum
from collections import namedtuple
import numpy as np

pygame.init()
font = pygame.font.Font('arial.ttf', 25)

# Definición de la enumeración para las direcciones
class Direction(Enum):
    DERECHA = 1
    IZQUIERDA = 2
    ARRIBA = 3
    ABAJO = 4

# Definición de la tupla Point para representar una posición en el juego
Point = namedtuple('Point', 'x, y')

# Gama de colores rgb
BLANCO = (255, 255, 255)
ROJO = (200, 0, 0)
AZUL1 = (0, 0, 255)
AZUL2 = (0, 100, 255)
NEGRO = (0, 0, 0)

TAM_BLOQUE = 20
VELOCIDAD = 40

# Clase SnakeGameAI que representa el juego de la serpiente
class SnakeGameAI:

    def __init__(self, w=640, h=480):
        self.w = w
        self.h = h
        self.display = pygame.display.set_mode((self.w, self.h))  # Crear la ventana del juego
        pygame.display.set_caption('Snake')  # Establecer el título de la ventana
        self.clock = pygame.time.Clock()  # Reloj para controlar la velocidad del juego
        self.reset()  # Inicializar el estado del juego

    def reset(self):
        # Estado inicial del juego
        self.direction = Direction.DERECHA

        self.head = Point(self.w / 2, self.h / 2)
        self.snake = [self.head,
                      Point(self.head.x - TAM_BLOQUE, self.head.y),
                      Point(self.head.x - (2 * TAM_BLOQUE), self.head.y)]

        self.score = 0
        self.food = None
        self._place_food()
        self.frame_iteration = 0

    def _place_food(self):
        # Colocar la comida en una posición aleatoria
        x = random.randint(0, (self.w - TAM_BLOQUE) // TAM_BLOQUE) * TAM_BLOQUE
        y = random.randint(0, (self.h - TAM_BLOQUE) // TAM_BLOQUE) * TAM_BLOQUE
        self.food = Point(x, y)
        if self.food in self.snake:
            self._place_food()

    def play_step(self, action):
        self.frame_iteration += 1

        # 1. Recopilar información del usuario
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                quit()

        # 2. Mover la serpiente
        self._move(action)
        self.snake.insert(0, self.head)

        # 3. Verificar si ha perdido
        reward = 0
        game_over = False
        if self.is_collision() or self.frame_iteration > 100 * len(self.snake):
            game_over = True
            reward = -10
            return reward, game_over, self.score

        # 4. Colocar nueva comida o simplemente moverse
        if self.head == self.food:
            self.score += 1
            reward = 10
            self._place_food()
        else:
            self.snake.pop()

        # 5. Actualizar la interfaz de usuario y el reloj
        self._update_ui()
        self.clock.tick(VELOCIDAD)

        # 6. Devolver la recompensa, el indicador de finalización del juego y la puntuación actual
        return reward, game_over, self.score

    def is_collision(self, pt=None):
        # Verificar si la serpiente ha colisionado con una pared o consigo misma
        if pt is None:
            pt = self.head
        # Límite de golpes
        if pt.x > self.w - TAM_BLOQUE or pt.x < 0 or pt.y > self.h - TAM_BLOQUE or pt.y < 0:
            return True
        # Se golpea a sí misma
        if pt in self.snake[1:]:
            return True
        return False

    def _update_ui(self):
        # Actualizar la interfaz de usuario
        self.display.fill(NEGRO)

        for pt in self.snake:
            pygame.draw.rect(self.display, AZUL1, pygame.Rect(pt.x, pt.y, TAM_BLOQUE, TAM_BLOQUE))
            pygame.draw.rect(self.display, AZUL2, pygame.Rect(pt.x + 4, pt.y + 4, 12, 12))

        pygame.draw.rect(self.display, ROJO, pygame.Rect(self.food.x, self.food.y, TAM_BLOQUE, TAM_BLOQUE))

        text = font.render("Puntuación: " + str(self.score), True, BLANCO)
        self.display.blit(text, [0, 0])
        pygame.display.flip()

    def _move(self, action):
        # Mover la serpiente en la dirección especificada por la acción

        # [enfrente, derecha, izquierda]
        sentido_agujas = [Direction.DERECHA, Direction.ABAJO, Direction.IZQUIERDA, Direction.ARRIBA]
        idx = sentido_agujas.index(self.direction)

        if np.array_equal(action, [1, 0, 0]):
            new_dir = sentido_agujas[idx]  # Sin cambio
        elif np.array_equal(action, [0, 1, 0]):
            next_idx = (idx + 1) % 4
            new_dir = sentido_agujas[next_idx]  # Girar a la derecha: derecha -> abajo -> izquierda -> arriba
        else:  # [0, 0, 1]
            next_idx = (idx - 1) % 4
            new_dir = sentido_agujas[next_idx]  # Girar a la izquierda: derecha -> arriba -> izquierda -> abajo

        self.direction = new_dir

        x = self.head.x
        y = self.head.y
        if self.direction == Direction.DERECHA:
            x += TAM_BLOQUE
        elif self.direction == Direction.IZQUIERDA:
            x -= TAM_BLOQUE
        elif self.direction == Direction.ABAJO:
            y += TAM_BLOQUE
        elif self.direction == Direction.ARRIBA:
            y -= TAM_BLOQUE

        self.head = Point(x, y)

