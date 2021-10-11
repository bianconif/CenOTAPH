import numpy as np
import cenotaph.basics.generic_functions as gf

#Create a random distribution of points in the [0,1] 3D cube
num_points = 100
points = np.random.rand(num_points, 3)
mass_distr = np.ones(num_points)

#Compute eigenvalues and eigenvectors via inertia matrix
_, eigvals_im, eigvecs_im = gf.inertia_matrix(points, mass_distr, central=True)

#Compute eigenvalues and eigenvectors via inertia matrix
eigvals_pca, eigvecs_pca = gf.pca(points)

a = 2