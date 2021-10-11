import math

import numpy as np

import cenotaph.combinatorics.transformation_groups as tg
from cenotaph.basics.generic_functions import convert_base
from cenotaph.combinatorics.necklaces_and_bracelets import find_orbits

"""Generic functions for histograms of equivalent patterns"""

def replace_words(words_in, old_dictionary, new_dictionary):
    """Replace any occurrence that appear in old dictionary with the
    corresponding word in the new dictionary
    
    Parameters
    ----------
    words_in : ndarray of int
        The input words as decimal integers
    old_dictionary : list (N) of int
        The old dictionary
    new_dictionary : list (N) of int
        The new dictionary
        
    Returns
    -------
    words_out : ndarray of int
        The words after the replacement
    """
    
    #The old and new dictionary must have the same number of elements
    if not len(old_dictionary) == len(new_dictionary):
        raise Exception("""The old and new dictionary must have the same 
                           number of elements""")
    
    #Do the replacements
    sort_idx = np.argsort(old_dictionary)
    idx = np.searchsorted(old_dictionary, words_in, sorter = sort_idx)
    words_out = np.asarray(new_dictionary)[sort_idx][idx]    
    
    return words_out

def group_invariant_dictionary(dictionary_in, num_colours, num_points,
                               group_action, **kwargs):
    """Given an input dictionary compute classes of equivalent words under
    the action of the group given
    
    Parameters
    ----------
    dictionary_in : ndarray of int
        The decimal codes of all the possible patterns
    num_colours : int
        The number of symbols that define the words in the dictionary - i.e.
        base of the decimal representation.
    num_points : int
        The number of elements of the set upon which the group acts (length
        of the words)
    group_action : str
        The group that acts on the patterns. Can be:
            'A' -> alternating group
            'C' -> cyclic group
            'D' -> dihedral group
            'S' -> symmetric group
    excluded_point (optional) : int
        Index of the point in the set excluded by the action of the group
            
    Returns
    -------
    dictionary_out : ndarray of int (same length as dictionary_in)
        Invariant labels under the action of the group given. 
        
    Note
    ----
    If group_action == 'A' or 'D' the invariant dictionary is computed via
    invariants, otherwise it it is computed via brute-force approach
    """
    
    #Generate the transformation group
    transformation_group = None
    if (group_action == 'A') or (group_action == 'S'):
        method = 'invariants'      
    elif (group_action == 'C') or (group_action == 'D'):
        method = 'brute-force'
    else:
        raise Exception('Group action not supported')
    
    #Unpack the decimal codes of the dictionary to generate the words
    words = convert_base(dictionary_in, num_colours)
    
    #Compute the orbits
    _, dictionary_out, _ = find_orbits(words, group_action, 
                                       method, **kwargs)
        
    #Return the group-invariant labels
    return dictionary_out