from math import gcd

import numpy as np
from scipy.special import comb
from sympy import totient

from cenotaph.combinatorics.necklaces_and_bracelets import find_orbits
from cenotaph.combinatorics.basics import get_divisors


#*** Check the definition of partitions and compositions: they can be messed up!

def _weak_compositions(n, k):
    """Generate compositions of n into k parts
    
    Parameters
    ----------
    n : int
        The input integer 
    k : int
        The number of parts
        
    Credits
    -------
    Sourced from: https://dandrake.livejournal.com/83095.html
    """
        
    if n < 0 or k < 0:
        return
    elif k == 0:
        # the empty sum, by convention, is zero, so only return something if
        # n is zero
        if n == 0:
            yield []
        return
    elif k == 1:
        yield [n]
        return
    else:
        for i in range(0,n+1):
            for comp in _weak_compositions(n-i,k-1):
                yield [i] + comp
                

def compositions(n, k, strong = True):
    """Compositions of n into k parts
    
    Parameters
    ----------
    n : int
        The input integer 
    k : int
        The number of parts
    strong : bool
        Select True for strong compositions (zero is not considered as a part);
        False for weak compositions (zero is considered as a part)
        
    Returns
    -------
    k_compositions : ndarray (N,k)
        A two-dimensional ndarray where each row represents one composition.
    """
    
    k_compositions = list(_weak_compositions(n, k))
    
    #Remove the compositions containing zeros if only strong compositions are
    #requested
    if strong:
        strong_compositions = list()
        for composition in k_compositions:
            if 0 not in composition:
                strong_compositions.append(composition)
        k_compositions = strong_compositions
                
    return np.array(k_compositions)

def cyclic_and_dihedral_compositions(n, k, strong = True):
    """Cyclic and dihedral compositions of n into k parts. A cyclic composition 
    of n of length k is an equivalence class consisting of all the linear 
    compositions of n with length k that can be obtained from each other 
    by a cyclic shift; a dihedral composition is an equivalence class of all 
    linear compositions that can be obtained from each other either by a cyclic 
    shift or a reversal of order [1]. 
    
    Parameters
    ----------
    n : int
        The input integer 
    k : int
        The number of parts
    strong : bool
        Select True for strong compositions (zero is not considered as a part);
        False for weak compositions (zero is considered as a part)
        
    Returns
    -------
    k_cyclic_compositions : ndarray (M,k)
        A two-dimensional ndarray where each row represents the representative
        of one cyclic composition.
    k_dihedral_compositions : ndarray (N,k)
        A two-dimensional ndarray where each row represents the representative
        of one dihedrale composition.
        
    References
    ----------
    [1] Hadjicostas, P. Cyclic, dihedral and symmetrical carlitz compositions 
        of a positive integer (2017) Journal of Integer Sequences, 20 (8), art. 
        no. 17.8.5
    """    
    
    #Generate the compositions first
    k_compositions = compositions(n, k, strong)    
    
    #Compute the cyclic compositions as the orbits under the action of Ck
    _, inv_labels, _ = find_orbits(words = k_compositions, 
                                   group_type = 'C', 
                                   method = 'brute-force')
    _, unique_indices, *_ = np.unique(inv_labels,
                                      return_index = True, 
                                      return_inverse = True)
    k_cyclic_compositions = k_compositions[unique_indices,:]    
    
    #Compute the dihedral compositions as the orbits under the action of Dk
    _, inv_labels, _ = find_orbits(words = k_compositions, 
                                   group_type = 'D', 
                                   method = 'brute-force')
    _, unique_indices, *_ = np.unique(inv_labels,
                                      return_index = True, 
                                      return_inverse = True)
    k_dihedral_compositions = k_compositions[unique_indices,:] 
    
    return k_cyclic_compositions, k_dihedral_compositions
    
def number_of_cyclic_and_dihedral_compositions(n, k):
    """Number of cyclic and dihedral string compositions of n into k parts. The
    calculation is based on the formulas given in [1].
    
    Parameters
    ----------
    n : int
        The input integer 
    k : int
        The number of parts
        
    Returns
    -------
    n_cyclic_compositions : int
        The number of cyclic compositions of n into k parts
    n_dihedral_compositions : int
        The number of dihedral compositions of n into k parts
        
    References
    ----------
    [1]  Knopfmacher, A., Robbins, N. Some properties of dihedral compositions
         (2013) Utilitas Mathematica, 92, pp. 207-220.
    """
    
    #Get the greatest common divisor between n and k
    gcd_n_k = gcd(n, k)
    
    #Get the divisors of gcd_n_k and convert them to integers
    divisors_gcd_n_k = np.array(list(get_divisors(gcd_n_k)), dtype = np.int)
    
    #Compute the number of cyclic compositions
    n_cyclic_compositions = 0
    for j in divisors_gcd_n_k:
        n_cyclic_compositions += totient(j) * comb(n // j, k // j, exact = True) 
    n_cyclic_compositions = n_cyclic_compositions // n
    
    #Compute the number of dihedral compositions
    n_dihedral_compositions = 0
    #---------- to be implemented ------------
    
    return n_cyclic_compositions, n_dihedral_compositions
    

def partitions(n, k, strong = True):
    """Partitions of n into k parts. Recall that a partition is a 
    composition in which the order of the parts is not significant. Therefore 
    partitions can be seen as the orbits of compositions under the action of 
    the symmetric group Sk.
    
    Parameters
    ----------
    n : int
        The input integer 
    k : int
        The number of parts
    strong : bool
        Select True for strong partitions (zero is not considered as a part);
        False for weak partitions (zero is considered as a part)
        
    Returns
    -------
    k_partitions : ndarray (N,k)
        A two-dimensional ndarray where each row represents one partition.
    """
    
    #Generate the compositions first
    k_compositions = compositions(n, k, strong)
    
    #Compute the partitions as the orbits under the action of Sk
    _, inv_labels, _ = find_orbits(words = k_compositions, 
                                   group_type = 'S', 
                                   method = 'invariants')
    _, unique_indices, *_ = np.unique(inv_labels,
                                      return_index = True, 
                                      return_inverse = True)
    k_partitions = k_compositions[unique_indices,:]
    
    return k_partitions