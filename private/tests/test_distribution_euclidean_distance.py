import numpy as np
import matplotlib.pyplot as plt
import scipy.spatial as sp_sp
from cenotaph.basics.statistics import *

n = 500                 #Number of points
d = 2                   #Dimension of the space
nbins = 50              #Number of bins used to estimate the distribution
dist_range = (0,10)     #The range of the distances to be investigated

#Generate the random points
points = np.random.normal(0.0, 1.0, (n, d))

#Compute the distance matrix
dm = sp_sp.distance_matrix(points, points, p=2)

#Get the upper triangle of the distance matrix
idx = np.triu_indices(n)
distances = dm[idx]

#Estimate the distance distribution
h, bin_edges = np.histogram(distances, nbins, dist_range, density=True)
x = (bin_edges[0:len(bin_edges)-1] + bin_edges[1:len(bin_edges)])/2

plt.bar(x, h, align='center', alpha=0.5)
#plt.xticks(x, join(map(str, x)))
plt.ylabel('Probability')
plt.xlabel('Distance')
#plt.title('Programming language usage')

#Compute the distance distribution through the formula
y = np.zeros(len(x))
for i, item in enumerate(x):
    y[i], _ = distribution_euclidean_distance(item, d)
    
plt.plot(x, y)

plt.show()
print(distances)
