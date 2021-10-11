import numpy as np
import scipy.special as sp

def distribution_euclidean_distance(R, k):
    """Distribution of the Euclidean distance between randomly distributed 
    Gaussian points in n dimensions
    
    Parameters
    ----------
    R : float [0, Inf]
        The distance between any two points
    k : int
        The dimension of the space
    
    Returns
    -------
    pdf : float 
        The probability of any two points to be separated by a distance R
    cdf : float
        The probablity of any two points to be separated by a distance up to R
    
    References
    ----------
    [1] Thirey, B. and Hickman, R., Distribution of Euclidean distances between 
        randomly distributed Gaussian points in n-space, arXiv:1508.02238v1
    
    """
    
    pdf = (2**(1-k) * np.exp(-(R**2)/4) * R**(k-1)) / sp.gamma(k/2)
    cdf = 1 - sp.gammaincc(k/2, (R**2)/4) / sp.gamma(k/2)
    
    return pdf, cdf