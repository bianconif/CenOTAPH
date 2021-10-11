import numpy as np
from cenotaph.basics.square_lattice import *

def digital_disc(radius, side, n=2):
    """Digital discs in two dimensions

    Parameters
    ----------
    radius : int
        The (integer) radius of the disc
    side : int
        Side length of the embedding square lattice
    n : int
        The exponent used to compute the distance from the center. Use 0 for
        Chebyshev distance, e.g.:
            n = 0 -> 'Chebyshev distance
            n = 1 -> 'cityblock' distance
            n = 2 -> 'Euclidean' distance

    Returns
    -------
    1) x coordinates of the points in the disc (numpy.ndarray)
    2) y coordinates of the points in the disc (numpy.ndarray)
    3) x coordinates of the points in the lattice (numpy.ndarray)
    4) y coordinates of the points in the lattice (numpy.ndarray)
    5) Boolean mask indicating which points of the lattice are in the disc (
    True = in the disc). Same length as output 1-4
    """

    #Create the square lattice where the disc is to be embedded
    x,y = square_lattice(side)

    #Compute the distance from the center (0,0) of each point
    if n == 1:
        distance = np.add(np.absolute(x),np.absolute(y))
    elif n == 0:
        distance = np.max(np.absolute(np.vstack((x.T,y.T))),axis=0)
    else:
        distance = np.add(x**n,y**n)**(1/n)

    #Avoid incongruences with the shape (no. of dimensions) of distance
    distance = distance.flatten()

    #Points are those within the distance from center
    in_disc = (distance <= radius)
    return x[in_disc], y[in_disc], x, y, in_disc

