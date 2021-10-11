from cenotaph.basics.neighbourhood import *

radius = 2
max_num_points = 10
full = True
nb = DigitalCircleNeighbourhood(radius, max_num_points, full)
nb.show()

#radius = 1
#ntype = 'Disc-L0'
#nb = Neighbourhood(radius, ntype)
#nb.show()

#Rotate
#angle = 20
#nb.rotate(angle)
#nb.show()

#Reflect around the horizontal axis
#A = 0
#B = 1
#C = 0
#nb.reflect(A, B, C)
#nb.show()

#Reflect around the x = y line
#A = 1
#B = -1
#C = 0
#nb.reflect(A, B, C)
#nb.show()

#Translate along the x axis
#angle = 20.0
#displacement = 1.0
#nb.translate(angle, displacement)
#nb.show()

#radius = 2
#nb = Neighbourhood(radius, 'Disc-L0')
#nb.show()

#side = 3
#nb = SquareNeighbourhood(side)
#nb.show()

#Test translation
#angle = 45.0
#displacement = 1.0
#nb.translate(angle, displacement)
#nb.show()

#Test reflection
#A = 1
#B = 1
#C = 0
#nb.reflect(A, B, C)
#nb.show()

#Test rotation
#angle = 90
#nb.rotate(angle)
#nb.show()

#nb.toroidal_wrap()
#nb.show()
