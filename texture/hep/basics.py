"""Generic functions and base classes for histograms of equivalent patterns"""

from abc import abstractmethod

import numpy as np

import cenotaph.combinatorics.transformation_groups as tg
from cenotaph.basics.base_classes import ImageDescriptor
from cenotaph.basics.generic_functions import convert_base, multilevel_thresholding
from cenotaph.basics.neighbourhood import DigitalCircleNeighbourhood
from cenotaph.combinatorics.necklaces_and_bracelets import find_orbits

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

class HEP(ImageDescriptor):
    """Base class for Histograms of Equivalent Patterns"""
    
    def __init__(self, radius=1, num_peripheral_points=8, is_full=True,
                 group_action=None, **kwargs):
        """Constructor

        Parameters
        ----------
        radius : int
            The radius of the digital circle.
        num_peripheral_points : int
            The number of peripheral points in the neighbourhood.
        is_full : bool
            Whether the base neighbourhood is full (i.e. contains the
            central pixel or not).
        group_action : str
            String indicating whether group-invariant features should be
            computed. Possible values are:
                'S' -> symmetric group
                'A' -> alternating group
                'C' -> cyclic group
                'D' -> dihedral group
        cache_folder (optional) : str
            Destination folder where to store the group-invariant lookup table.
            If None the lookup table is computed again each time the
            descriptor is instantiated.
        """

        super().__init__()
        self._radius = radius
        self._num_peripheral_points = num_peripheral_points
        self._is_full = is_full
        self._group_action = group_action
        self._generate_neighbourhood()
        
        if (group_action is not None) and\
           group_action not in ['S', 'A', 'C',  'D']:
            raise Exception('Group action not supported')
        
        if 'cache_folder' in kwargs.keys():
            self._cache_folder = kwargs['cache_folder']

    def _generate_neighbourhood(self):
        if self.is_full():
            num_points = self._get_num_peripheral_points() + 1
        else:
            num_points = self._get_num_peripheral_points()
            
        self._neighbourhood = DigitalCircleNeighbourhood(self._radius,  
                                                         num_points,
                                                         full = self.is_full())         

    def is_full(self):
        """Whether the neighbourhood is of type 'full' or not"""
        return self._is_full

    def _get_num_peripheral_points(self):
        return self._num_peripheral_points
            
    def _get_num_points(self):
        retval = self._get_num_peripheral_points()
        if self.is_full():
            retval = retval + 1
        return retval
    
    def _get_weights(self):
        return self._weights 
        
    @abstractmethod
    def _get_pattern_maps(self):
        """Returns the pattern maps
        
        Returns
        -------
        pattern_maps: list of ndarrays of int (H,W)
            The class-specific pattern maps. Grey-scale methods return one
            pattern maps; colour methods can return more than one (intra- and
            inter-channel features)
        """
    
    @staticmethod
    def _compute_invariant_patterns():
        """Substitutes the original patterns with action-invariant ones
        
        Parameters
        ----------
        
        """
        
    @staticmethod
    def _compute_histogram(pattern_map, dictionary):
        """Compute the normalised histogram of the pattern map over a given
        dictionary.
        
        Parameters
        ----------
        pattern_map: ndarray of int (H,W)
            The pattern map.
        dictionary: ndarray of int (D)
            The dictionary.
            
        Returns
        -------
        histogram: ndarray of float (D)
            The probability of occurrence of each word of the dictionary in the
            pattern map
        """
        
        #Define the histogram hedges based on the dictionary
        bin_edges = np.append(dictionary, np.max(dictionary) + 1)
        
        #Compute the first-order statistics over the pattern map
        histogram, _ = np.histogram(pattern_map.flatten(), 
                                    bin_edges, 
                                    density=True)    
        
        return histogram
    
    def _compute_features(self):
        
        features = np.array([])
          
        #Compute the pattern maps
        pattern_maps = self._get_pattern_maps()
        
        #Get the dictionary and compute the bin edges
        dictionary = self._get_dictionary()
        
        for pattern_map in pattern_maps:
            
            patterns = pattern_map.flatten()
                    
            #Manage invariance to group actions
            if self._group_action is not None:
                invariant_dictionary = self._get_invariant_dictionary()
                invariant_patterns = replace_words(patterns, 
                                                   dictionary, 
                                                   invariant_dictionary)
                #Rebind the variables
                dictionary = list(set(invariant_dictionary))
                patterns = invariant_patterns
        
            features_ = self._compute_histogram(patterns, dictionary)
            features = np.hstack((features, features_))
                
        return features
    
    def _generate_patterns_by_thresholding(self, data_in, pivot, thresholds, 
                                           equalities, weights):
        """Compute patterns based on thresholding
        
        Parameters
        ----------
        data_in : ndarray of int or float (H,W,L)
            The input data - i.e the image patterns. H, W
            are the height and width of the image; L is the number of points
            to be compared with pivot.
        pivot : ndarray of int or float (H,W). 
            The values against which the input data are to be compared.
        thresholds : list of int or float
            The thresholds used for thresholding the difference 
            (data_in - pivot)
        equalities : bool
            Whether to consider specific levels when the input data are exactly
            equal to the threshold values. This is for instance False for LBP
            and True for TS.
        weights : list of int
            List of weights used to generate the patterns
            
        Returns
        -------
        patterns : ndarray of int (H,W)
            The map of patterns as decimal codes
        """         
        
        #Create an empty container for the comparisons
        comparisons = np.zeros(data_in.shape, dtype = 'int')
        
        #Broadcast the pivot values
        pivot_tiled = np.repeat(pivot[:, :, np.newaxis], 
                                data_in.shape[-1], 
                                axis = -1)
        #np.tile(pivot, (1,1,data_in.shape[-1]))
        
        #Create the mask      
        mask = self._create_mask(comparisons.shape[0], 
                                 comparisons.shape[1], 
                                 weights)  
        #Compute the differences and do the thresholding
        differences = np.subtract(data_in, pivot_tiled)
        comparisons = multilevel_thresholding(differences, 
                                              thresholds,
                                              self._consider_equalities())
                       
        #Generate and return the patterns      
        patterns = np.multiply(comparisons, mask)
        patterns = np.sum(patterns, axis=2)
        
        return patterns    
    
    @staticmethod
    def _create_mask(height, width, weights):
        """Generates the multiplication mask for pattern numbering
        
        Parameters
        ----------
        width, height : int
            The dimension of the pattern map.
        pattern_length : int
            The pattern length (number of characters 'beads' which
            make up a pattern).
        weights : list of int
            The multiplication weights. 
        
        Returns
        -------
        mask : int (height, width, num_weights)
            The multiplication mask
        """
        mask = np.tile(weights, (height,width,1))
        return mask  
    
    @abstractmethod
    def _get_pattern_maps(self):
        """Generate the pattern map. This is the 'kernel function'
        as defined in Fernández et al. 2013.
                
        Returns
        -------
        patterns : ndarray of int (H,W,D) 
            The pattern map, where (H,W) are the height and width of the
            input image; D the pattern length.
        """
    
    @abstractmethod
    def _get_dictionary(self):
        """Dictionary of texton codes generated by the descriptor
        
        Returns
        -------
        dictionary : ndarray of int
            The dictionary of texton codes as decimal integers
        """
    
    @abstractmethod
    def _get_num_colours(self):
        """Number of colours (symbols) used for pattern encoding
        
        Returns
        -------
        num_colours : int
            The number of colours
        """
    
    def _get_name_of_invariant_dictionary(self):
        """Conventional name for the group-invariant dictionary of texton codes 
        generated by the descriptor. Used for caching the dictionary.
    
        Returns
        -------
        name : str
            Conventional name for the group-invariant dictionary 
        """
        retval = self.__class__.__name__\
            + '-' + self._group_action\
            + '-' + str(self._get_num_peripheral_points()) + 'beads'\
            + '-' + str(self._get_num_colours()) + 'colours'
        return retval
        
        
    def _get_invariant_dictionary(self):
        """Return the dictionary of texton codes invariant under a 
        group action
        
        Returns
        -------
        invariant_dict : list of int
            The group-invariant dictionary
        """

        try:
            source = self._cache_folder + '/'\
                + self._get_name_of_invariant_dictionary()\
                + '.txt'
            #If a cache folder is provided check if the dictionary is stored 
            #in the cache; if so, load it.            
            if os.path.isfile(source):
                invariant_dict = list()
                with open(source) as in_file:
                    record = None
                    while record != "":
                        record = in_file.readline()
                        record = record.strip('\n')
                        fields = record.split(',')
                        try:
                            invariant_dict.append(int(fields[1]))
                        except:
                            break
            #Otherwise compute and store tye dictionary
            else:
                invariant_dict = self._compute_invariant_dictionary()
                original_dict = self._get_dictionary()
                with open(source, 'w') as out_file:
                    for _, i in enumerate(original_dict):
                        out_file.write('{:d},{:d}\n'.format(original_dict[i],
                                                            invariant_dict[i]))
                
        #If no cache folder is given just compute the dictionary without 
        #storing it        
        except AttributeError:          
            invariant_dict = self._compute_invariant_dictionary()
        
        return invariant_dict
        
    @abstractmethod
    def _compute_invariant_dictionary(self):
        """Compute dictionary of texton codes invariant under a group action
        
        Returns
        -------
        dictionary : ndarray of int
            The dictionary of texton codes invariant under group action 
            as decimal integers
        """
        
    def __repr__(self):
        retval = self.__class__.__name__\
            + '-r' + str(self._radius)\
            + '-n' + str(self._num_peripheral_points)
        if self._group_action is None:
            retval = retval + '-gNone'
        else:
            retval = retval + '-g' + self._group_action
        
        return retval     