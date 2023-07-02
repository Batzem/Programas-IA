from pyamaze import maze,agent

#-----------------------------------------
#    Crear laberinto
#    x,y posición de la meta
#-----------------------------------------
m= maze(25,40)
m.CreateMaze(x=1, y=1)

#----------------------------------
# Poner agente en el laberinto
#-----------------------------------
a = agent(m,footprints=True,filled=True)

#---------------------------------------
#  Graficar la trayectoria del agente
#---------------------------------------
m.tracePath({a:m.path},delay=25
m.run()
