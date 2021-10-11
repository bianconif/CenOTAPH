import numpy as np

def square_lattice(side):
    """Integer square lattice in two dimensions centered on (0,0)

    Parameters
    ----------
    side : int
        An integer odd number representing the length of the side of the
        lattice. If an even number is given this is transformed to the
        successive odd integer.

    Returns
    -------
    1) x coordinates of the points in the lattice (numpy.ndarray)
    2) y coordinates of the points in the lattice (numpy.ndarray)
    """

    #Round side to the upper odd integer
    side = round(side)
    if (side % 2) == 0:
        side = side + 1

    #One-dimensional generator
    generator = np.arange(0,side)

    #Center the generator
    generator = generator - (side - 1)/2

    #Create the square lattice
    lattice = np.meshgrid(generator, generator)
    x = lattice[0]
    y = lattice[1]

    x = np.reshape(x, (x.size,1))
    y = np.reshape(y, (y.size,1))

    return x,y