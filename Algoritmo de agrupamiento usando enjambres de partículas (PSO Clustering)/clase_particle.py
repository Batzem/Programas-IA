#----------------------------
# ESFM IPN FUNDAMENTOS DE IA
# TONATIUH HERNANDEZ
#----------------------------

#-----------------
# Clase particle
#-----------------
import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

class Particle:
  def __init__(self, n_clusters, data, use_Kmeans=True, w=0.72, c1=1.49, c2=1.49):
      self.n_clusters = n_clusters
      if use_Kmeans:
        k_means = KMeans(n_clusters=self.n_clusters)
        k_means.fit(data)
        self.centroids_pos = k_means.cluster_centers_
      else:
        self.centroids_pos = data[np.random.choice(list(range(len(data))), self.n_clusters)]


        #------------------------------------------------------------------------------
        # Cada agrupamiento tiene un centroide que es el punto  que lo representa
        # se asignan  k datos aleatorios  a k centroides 
        #------------------------------------------------------------------------------
        self.pb_val = np.inf

        #-------------------------------------------------------------------
        #   Mejor posición personal  para todos los centroides hasta aquí
        #-------------------------------------------------------------------
        self.pb_pos = self.centroids_pos.copy()
        self.velocity = np.zeros_like(self.centroids_pos)

        #---------------------------------------------
        # Mejor agrupamiento de los datos hasta aquí
        #----------------------------------------------
        self.pb_clustering = None

        #---------------------------------------------------------------------
        # Parámetros del PS0 (particle_swarm_optimization)
        #                     (optimización usando enjambres de partículas)
        #---------------------------------------------------------------------
        self.w = w
        self.c1 = c1
        self.c2 = c2

def update_pb(self, data: np.ndarray):
  """
  Actualiza el mejor puntaje basado en la función de aptitud (Ecuación 4)
  """
  #---------------------------------------------------------------------
  # Encuentra los datos (puntos) que pertenecen a cada agrupamiento 
  # utilizando distancias a los centroides 
  #---------------------------------------------------------------------
  distances = self._get_distances(data = data)

  #--------------------------------------------------------------------------
  # La distancia mínima entre los datos y un centroide indica que pertenece 
  # a ese agrupamiento 
  #---------------------------------------------------------------------------
  clusters = np.argmin(distances, axis = 0) #shape: (len(data),)
  clusters_ids = np.unique(clusters)

  #-------------------------------------------------------------------------------
  # Si el algoritmo genera menos de n agrupamientos generar al azar la posición 
  # de un nuevo centroide para el id del agrupamiento que falta
  #---------------------------------------------------------------------------------
  while len(clusters_ids) != self.n_clusters:
    deleted_clusters = np.where(np.isin(np.arange(self.n_clusters), clusters_ids) == False)[0]
    self.centroids_pos[deleted_clusters] = data[np.random.choice(list(range(len(data))), len(deleted_clusters))]
    distances = self.get_distances(data=data)
    clusters = np.argmin(distances, axis=0)
    clusters_ids = np.unique(clusters)

  new_val = self.fitness_function(clusters=clusters, distances=distances)
  if new_val < self.pb_val:
    self.pb_val = new_val
    self.pb_pos = self.centroids_pos.copy()
    self.pb_clustering = clusters.copy()

  def update_velocity(self, gb_pos: np.ndarray):
    '''
    Actualiza la velocidad usando la anterior, la mejor posición personal y la mejor posición
    en el enjambre
      :param gb_pos: vector de las mejores posiciones de los centroides entre todas las partículas
    '''
    self.velocity = self.w * self.velocity + \
                    self.c1 * np.random.random() * (self.pb_pos - self.centroids_pos) + \
                    self.c2 * np.random.random() * (gb_pos - self.centroids_pos)

   def move_centroids(self, gb_pos):
       self.update_velocity(gb_pos=gb_pos)
       new_pos = self.centroids_pos + self.velocity
       sel.centroids_pos = new_pos.copy()
   
   def _get_distances(self, data: np.ndarray) -> np.ndarray:
    '''
      Calcula la distancia euclidiana entre los datos y los centroides
      :param data:
      :return: distancias: un arreglo de numpy de las distancias (len(centroids) x len(data))
      '''
      distances = []
      for centroid in self.centorids_pos:
        #Calcula distancia euclidiana --> raiz de la suma de los cuadrados
        d = np.linalg.norm(data - centroid, axis=1)
        distances.append(d)
      distances = np.array(distances)
      return distances

   def fitness_function(self,clusters: np.ndarray, distances : np.ndarray) -> float:
      """ 
      Evalúa la funcion de aptitud ( Ecuación 4)
      i es el índice de la partícula
      j es el índice de los agrupamientos de la particula i
      p es el vector de los indices de los datos de entrada en el agrupamiento [ij]
      z[p] es el vector de los datos de entrada en el agrupamiento [ij]
      d es el vector de las distancias entre z(p) y el centroide j
      :param clusters:
      :param distances:
      :return J:
      """
      J = 0.0
      for i in range(self.n_clusters):
        p = np.where(clusters == i)[0]
        if len(p):
          d = sum(distances[i][p])
          d /= len(p)
          J += d
      J / = self.nclusters
      return J
#-----------------------------------------------------------------------------------------------------------
