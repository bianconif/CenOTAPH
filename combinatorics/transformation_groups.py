"""Basic functions for the combinatorics of necklaces and bracelets"""

from abc import abstractmethod, ABC

import numpy as np
from sympy.combinatorics.permutations import Permutation
from sympy.combinatorics.generators import alternating, symmetric

from cenotaph.third_parties.doc_inherit import doc_inherit

class TransformationGroup(ABC):
    """Base class for generic transformation group"""
    
    def __init__(self, n, include_id = False):
        """The constructor. Generates the elements of the group (by default
        the identity is not included).
        
        Parameters
        ----------
        n : int
            The order (number of elements of the group)
        include_id : bool (optional)
            Whether to include the identity among the elements of the group.
        """         
        
        self._n = n        
        self._include_id = include_id
        
        #Define the elements of the group
        self._elements = list()
        
    def act_on(self, data_in):
        """Group action on a set of elements
        
        Parameters
        ----------
        data_in : ndarray (M,N)
            The input data - i.e. the set of elements which the group acts on.
            Each row represents a different element of the set.
        
        Returns
        -------
        data_out : ndarray (M,N,R)
            The data after the action of the group. The results of the 
            action of each element ri, i = {0,...,R-1} of the group 
            are stacked on the second axis. By convention the first element
            (i = 0) represents the identity (if present - see parameter
            include_id in the constructor of the subclasses). 
        """
        
        data_out = np.copy(data_in)
        data_out = np.expand_dims(data_out, axis = -1)


        #Let the elements of the group act on the input data
        for elem in self._elements:
            result = elem(data_in)
            result = np.expand_dims(result, axis = -1)
            data_out = np.concatenate((data_out, result), axis = -1)
            
        return data_out
    
    @staticmethod    
    def factory(group_type, order, include_id=False):
        """Factory to generate transformation groups

        Parameters
        ----------
        group_type : str 
            Label (name) indicating the group to be generated. Supported
            values are:
                'A' -> Alternating group
                'C' -> Cyclic group
                'D' -> Dihedral group
                'S' -> Symmetric group
        order : int
            The order (number of elements) of the group.
        include_id : bool
            Whether to include the identity.
        """
        retval = None
        
        if group_type == 'A':
            retval = AlternatingGroup(order, include_id)
        elif group_type == 'C':
            retval = CyclicGroup(order, include_id)    
        elif group_type == 'D':    
            retval = DihedralGroup(order, include_id)
        elif group_type == 'S':
            retval = SymmetricGroup(order, include_id)
        else:    
            raise Exception('Group type unknown/unsupported')
    
        return retval    
    
    def __repr__(self):
        return self.__class__.__name__
    
class SymmetricGroup(TransformationGroup):
    """Symmetric group of order n"""
    
    @staticmethod
    def _generate_elements(group_type, n=4, include_id=False):
        """Generates the elements of the symmetric or alternating group
        
        Parameters
        ----------
        group_type : str
            The group to generate. Can be:
                'symmetric'   -> the symmetric group Sn
                'alternating' -> the alternating group An
        n : int
            The order of the group
        include_id : bool
            Whether to include the identity element
                
        Returns
        -------
        group_elements : list of functions
            The elements of the group
        """
        
        #Generate the permutations via sympy.combinatorics.generators
        permutations = None
        if group_type == 'symmetric':
            permutations = symmetric(n)
        elif group_type == 'alternating':
            permutations = alternating(n)
        else:
            raise Exception('Group type not supported')
            
        #Make functions out of the permutations
        group_elements = list()
        for permutation in permutations:
            group_elements.append(_make_permutation(permutation))
                        
        if not include_id:
            group_elements.pop(0)        
        
        return group_elements
            
    @doc_inherit
    def __init__(self, n, include_id = False):
        super().__init__(n = n, include_id = include_id)
        self._elements = self._generate_elements("symmetric", 
                                                 n = self._n, 
                                                 include_id = self._include_id)

class AlternatingGroup(SymmetricGroup):
    """Alternating group of order n"""
    
    @doc_inherit
    def __init__(self, n, include_id = False):
        super().__init__(n = n, include_id = include_id)
        self._elements = self._generate_elements("alternating", 
                                                 n = self._n, 
                                                 include_id = self._include_id) 

class CyclicGroup(TransformationGroup):
    """Cyclic group of order n"""
    
    @doc_inherit
    def __init__(self, n, include_id = False):
        super().__init__(n = n, include_id = include_id)
        
        if include_id:
            counter = range(0, n)
        else:
            counter = range(1, n)
        
        for i in counter:
            self._elements.append(_make_circular_shift(i))
 
class DihedralGroup(CyclicGroup):
    """Dihedral group of order n"""
    
    @doc_inherit
    def __init__(self, n, include_id = False):
        
        #Generate the elements of the cyclic group
        super().__init__(n = n, include_id = True)
        
        #Add the reflections
        reflections = list()
        for elem in self._elements:
            reflections.append(make_mirror(elem))
        self._elements = self._elements + reflections
        
        #Remove the identity if not required
        if not include_id:
            self._elements.pop(0)
                     
def _make_permutation(perm):
    """Generate a function that implements a permutation
    
    Parameters
    ----------
    perm : Permutation (sympy.combinatorics.permutations)
        The permutation object
        
    Returns
    ----------
    permutation : function
        The permutation function
    """
    
    def permutation(data_in):
        data_out = [perm(data_in[i,:]) 
                    for i in range(data_in.shape[0])]
        data_out = np.array(data_out)
        return data_out
    
    return permutation
    
        
def _make_circular_shift(shift):
    """Generate a function that implements a circular shift
    
    Parameters
    ----------
    shift : int
        The entity of the shift
    
    Returns
    -------
    circular_shift : function
        The circular shift function
    """
    
    def _circular_shift(data_in):
        """
        Parameters
        ----------
        data_in : ndarray (M,N)
            The input data, to be circularly shifted columnwise
        
        Returns
        -------
        data_out : ndarray (M,N)
            The circularly-shifted matrix
        """
        return np.roll(data_in, shift, axis = 1)   
    return _circular_shift

def make_mirror(fun_in):
    """Generate a function that implements a mirroring of the input sequence
    
    Parameters
    ----------
    fun_in : function
        The inner function the mirror is applied to (circular shift in
        this case)
        
    Returns
    -------
    mirror : function
        The mirror function
    """
        
    def mirror(data_in):
        """
        Parameters
        ----------
        data_in : ndarray (M,N)
            The input data, to be circularly shifted columnwise
            
        Returns
        -------
        data_out : ndarray (M,N)
            The matrix mirrored in columnwise direction (i.e. around the
            middle vertical axis)
        """
        
        return np.flip(fun_in(data_in), axis = 1)
    return mirror
