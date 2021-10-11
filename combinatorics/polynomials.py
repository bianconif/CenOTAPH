from itertools import combinations

import numpy as np

def symmetric_polynomial(X, d):
    """Symmetric polynomial of degree d over X. For example, given an input
    vector X = {x1,...xn}, we have:
    S(X,1) = x1 + ... + xn
    S(X,2) = x1x2 + x2x3 + ... xn-1xn
    ...
    S(X,n) = x1x2x3...xn
    
    
    Parameters
    ----------
    X : ndarray of int or float (P,N)
        The data on which the polynomial is computed.
    d : int
        The degree of the polynomial.
        
    Returns
    -------
    S : ndarray of int or float (P)
    """
    
    #Create an empty array
    S = np.zeros(X.shape[0], dtype = X.dtype)
    
    #Compute the list of all subsets of X of length d
    combs = combinations(range(X.shape[1]), d + 1) 
    
    #Compute the polynomial
    for comb in combs:
        S = S + np.prod(X[:,comb], axis = 1)
        
    return S

def symmetric_polynomials(X):
    """All the symmetric polynomials over X
    
    Parameters
    ----------
    X : ndarray of int or float (P,N)
        The data on which the polynomial is computed.
        
    Returns
    -------
    S : ndarray of int or float (P,N)
        Each column j represents the symmetric polynomial of degree j + 1 
    """
    
    #Create an empty array
    S = np.zeros(X.shape, dtype = X.dtype)    
    
    for d in range(X.shape[1]):
        S[:,d] = symmetric_polynomial(X, d)
        
    return S

def vandermonde_polynomial(X):
    """Vandermonde polynomial over X
    
    Parameters
    ----------
    X : ndarray of int or float (P,N)
        The data on which the polynomial is computed.
        
    Returns
    -------
    S : ndarray of int or float (P)
        Each i-th element represents the Vandermonde polynomial of X[i,:] 
    """   
    
    #Create an empty array
    S = np.ones(X.shape[0], dtype = X.dtype)   
    
    #Compute Vandermonde product
    for j in range(1, X.shape[1]):
        for i in range(0, j):
            S = np.multiply(S, X[:,j] - X[:,i])
      
    return S
        
    
        
    