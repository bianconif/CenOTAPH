import matplotlib.pyplot as plt
import numpy as np
# from basics.square_lattice import *
# from basics.digital_disc import *
# from basics.digital_circle import *
# from basics.matrix_displaced_copies import *

from cenotaph.basics.generate_neighbourhood import generate_neighbourhood
from cenotaph.basics.digital_disc import digital_disc

radius = 3
ntype = 'Andres-L0'
neighbourhood = generate_neighbourhood(radius, ntype) 
x = neighbourhood[:,0]
y = neighbourhood[:,1]
plt.plot(x,y, marker='.', color='k', linestyle='none')
plt.title('Neighbourhood type = ' + ntype + '; radius = ' + str(radius))
plt.show()

#radius = 2
#side = 7
#n = 0
#xd,yd,xs,ys,_ = digital_disc(radius, side, n)
##
#plt.plot(xs,ys, marker='+', color='k', linestyle='none')
#plt.plot(xd,yd, marker='.', color='r', linestyle='none')
#plt.title('Digital disc (radius = ' + str(radius) + '; n = ' + str(n) + ')')
#plt.show()

#dc = digital_circle(radius, n)

# plt.plot(dc[:,0],dc[:,1], marker='.', color='k', linestyle='none')
# for r in range(len(dc[:,0])):
#     plt.text(dc[r,0], dc[r,1], str(r), fontsize=12)
#
# plt.title('Digital circle (radius = ' + str(radius) + '; n = ' + str(n) + ')')
# plt.show()

# radius = 3
# n = 2
# disc = ds.digital_disc(radius, n)
# circle = ds.digital_circle(radius, n)
# #print(disc);
#
# plt.plot(disc[3],disc[4], marker='.', color='k', linestyle='none')
# plt.title('Digital disc (radius = ' + str(radius) + '; n = ' + str(n) + ')')
# plt.show()
#
# plt.plot(circle[0],circle[1], marker='.', color='k', linestyle='none')
# plt.title('Digital circle (radius = ' + str(radius) + '; n = ' + str(n) + ')')
# plt.show()

#M = np.mat('1 2 3; 4 5 6; 7 8 9')
#disps = np.mat('0 1; 1 0; 1 1')
#filtering = 'mirror'
#print(M)

#MD = matrix_displaced_copies(M, disps, filtering)
#print(MD)