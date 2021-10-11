"""Histograms of equivalent patterns: grey-scale versions"""

from abc import abstractmethod
import os

import numpy as np

from cenotaph.basics.base_classes import ImageDescriptor, ImageDescriptorGS
from cenotaph.basics.generic_functions import multilevel_thresholding
from cenotaph.basics.neighbourhood import *
from cenotaph.basics.matrix_displaced_copies import matrix_displaced_copies
from cenotaph.texture.hep.basics import *

from cenotaph.third_parties.doc_inherit import doc_inherit

import matplotlib.pyplot as plt

class HEPGS(ImageDescriptorGS):
    """Abstract base class for grey-scale HEP descriptors. Implementation
    based on digital circkle neighbourhoods"""

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
            central pixel or not)
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
                
    @staticmethod  
    def factory(method, radius=1, num_peripheral_points=8,
                group_action=None, **kwargs):
        """Generate grey-scale HEP descriptors
        
        Parameters
        ----------
        method : str
            The name of the descriptor to be created. Can be:
                'LBP' -> Local Binary Patterns.
        radius : int
            The radius of the digital circle.
        num_peripheral_points : int
            The number of peripheral points in the neighbourhood.
        group_action : str or None
            String indicating whether group-invariant features should be
            computed. Possible values are:
                'S' -> symmetric group
                'A' -> alternating group
                'C' -> cyclic group
                'D' -> dihedral group
            Use None for no action.
        cache_folder (optional) : str
            Destination folder where to store the group-invariant lookup table.
            If None the lookup table is computed again each time the
            descriptor is instantiated.
        """
               
        descriptor = None
        if method == 'LBP':
            descriptor = LBP
        elif method == 'LTP':
            descriptor = LTP
        elif method == 'TS':
            descriptor = TS   
        else:
            raise Exception('Unrecognised descriptor')
        
        return descriptor(radius = radius, 
                          num_peripheral_points = num_peripheral_points, 
                          group_action = group_action,
                          **kwargs)
    
    
    #@doc_inherit
    def _compute_features(self):
        """Compute the histogram of equivalent patterns
        
        Parameters
        ----------
        img : Image
            The input image
        """
        
        features = np.array([])
        
        #Generate displaced copies of the input image
        self._img_layers = matrix_displaced_copies(self._img_in.get_data(), 
                                                   self._neighbourhood.\
                                                   get_integer_points())  
        #Compute the pattern map
        patterns = self._compute_patterns()
        
        #Get the dictionary and compute the bin edges
        dictionary = self._get_dictionary()
        
        #Manage invariance to group actions
        if self._group_action is not None:
            invariant_dictionary = self._get_invariant_dictionary()
            invariant_patterns = replace_words(patterns, 
                                               dictionary, 
                                               invariant_dictionary)
            #Rebind the variables
            dictionary = list(set(invariant_dictionary))
            patterns = invariant_patterns
        
        #Define the histogram hedges based on the dictionary
        bin_edges = np.append(dictionary, np.max(dictionary) + 1)
        
        #Compute the first-order statistics over the pattern map
        features, _ = np.histogram(patterns, 
                                   bin_edges, 
                                   density=True)
        
        return features
    
    @abstractmethod
    def _compute_patterns(self):
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
    
    #@staticmethod
    def _create_mask(self, height, width, weights):
        """Multiplication mask for pattern numbering
        
        Parameters
        ----------
        width, height : int
            The dimension of the pattern map
        pattern_length : int
            The pattern length (number of characters 'beads' which
            make up a pattern)
        weights : list of int
            The multiplication weights 
        
        Returns
        -------
        mask : int (height, width, num_weights)
            The multiplication mask
        """
        mask = np.tile(weights, (height,width,1))
        return mask
        
    def __repr__(self):
        retval = self.__class__.__name__\
            + '-r' + str(self._radius)\
            + '-n' + str(self._num_peripheral_points)
        if self._group_action is None:
            retval = retval + '-gNone'
        else:
            retval = retval + '-g' + self._group_action
        
        return retval    

class HEPGSLocalThresholding(HEPGS):
    """HEP methods based on local thresholding - e.g. LBP, ILBP, TS,
    LTP, etc."""
    
    @doc_inherit
    def __init__(self, radius=1, num_peripheral_points=8, group_action=None, 
                 **kwargs):
        super().__init__(radius = radius, 
                         num_peripheral_points = num_peripheral_points, 
                         is_full = True,
                         group_action = group_action, **kwargs)

        #Generate the weights or defining the dictionary
        self._weights = self._get_num_colours() **\
            np.arange(self._num_peripheral_points) 
    
    @abstractmethod
    def _get_pivot(self):
        """Compute the pivot value used for thresholding. This can be,
        for instance, the value of one specific point in the neighbourhood
        or the average of all the points
        
        Returns
        -------
        pivot : ndarray of int or float (H,W)
        """
        
    @abstractmethod
    def _get_base_values(self):
        """Compute the base values which will be compared with the pivot
        values. These can be, for instance, the values of the peripheral
        points in the neighbourhood or of all the points
        
        Returns
        -------
        base_values : ndarray of int or float (H,W,L)
        """
        
    @abstractmethod
    def _consider_equalities(self):
        """Whether equalities define different levels in the thresholding
        step. For instance, if there is just one threshold value, say t = 0,
        and an input value x the thresholding will produce the following 
        results:
            level = 0 if x <= 0
            level = 1 if x > 0
        -- if the returned value is False --
        and 
            level = 0 if x <= 0
            level = 1 if x = 0
            level = 2 of x >= 0
        -- if the returned value is True --
        """

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
        
    def _compute_patterns(self):
        patterns = self._generate_patterns_by_thresholding(
            self._get_base_values(), self._get_pivot(), 
            self._get_thresholds(), self._consider_equalities(),
            self._get_weights())  
        
        #fig, (ax1, ax2) = plt.subplots(nrows=2)
        #ax1.imshow(self._img_in.get_data(), cmap=plt.get_cmap('Greys'))
        #ax1.set_title('Original image')
        
        #ax2.imshow(patterns)
        #ax2.set_title('Patterns')        
        
        #plt.show()
        #a = 0
        return patterns
         
        
class LBP(HEPGSLocalThresholding):
    """Local binary patterns
    
    References
    ----------
    [1] Ojala, T., Pietikainen, M., Maenpaa, T.
        Multiresolution gray-scale and rotation invariant texture 
        classification with local binary patterns (2002) IEEE Transactions
        on Pattern Analysis and Machine Intelligence, 24 (7), pp. 971-987
    """
    
    @doc_inherit
    def _get_pivot(self):
        return self._img_layers[:,:,self._neighbourhood.center_index()]

    @doc_inherit
    def _get_base_values(self):
        return self._img_layers[:,:,self._neighbourhood.peripheral_indices()]
                        
    def _get_dictionary(self):
        return list(range(self._get_num_colours() **\
                          self._get_num_peripheral_points()))
    
    def _get_thresholds(self):
        """Values for thresholding the difference between the peripheral
        and central pixel
        """
        return [0]  
    
    @doc_inherit
    def _consider_equalities(self):
        return False
    
    @doc_inherit
    def _get_num_colours(self):
        return 2
    
    @doc_inherit
    def _compute_invariant_dictionary(self):
        retval = None
        if self._group_action is not None:
            retval =\
                group_invariant_dictionary(self._get_dictionary(),
                                           num_colours = self._get_num_colours(),
                                           num_points = self._get_num_peripheral_points(),
                                           group_action = self._group_action) 
        return retval
    
    def __repr__(self):
        return super().__repr__()
    
class TS(LBP):
    """Texture spectrum
    
    References
    ----------
    [1] He, D.-C., Wang, L.
        Texture Unit, Texture Spectrum, and Texture Analysis
        (1990) IEEE Transactions on Geoscience and Remote Sensing, 28 (4), 
        pp. 509-512.
    """
    
    @doc_inherit
    def _consider_equalities(self):
        return True
    
    @doc_inherit
    def _get_num_colours(self):
        return 3
        
    def __repr__(self):
        return super().__repr__()
    
class LTP(LBP):
    """Local ternary patterns
    
    References
    ----------
    [1] Tan, X., Triggs, B.
        Enhanced local texture feature sets for face recognition under 
        difficult lighting conditions
        (2010) IEEE Transactions on Image Processing, 19 (6), art. no. 5411802, 
        pp. 1635-1650.
    """
    
    def _get_thresholds(self, mode = 'range', fraction = 0.02):
        """
        Parameters
        ----------
        mode : str
            The modality by which the threshold values are computed. Can be:
                'range' -> thresholds are computed as follows
                           t0 = -fraction * 2^bit_depth
                           t1 = fraction * 2^bit_depth
        fraction : float ]0,1[
            The multiplication factor defining the threshold values         
        """
        return [-fraction * 2**self._img_in.get_bit_depth(),
                fraction * 2**self._img_in.get_bit_depth()]
    
    @doc_inherit
    def _get_num_colours(self):
        return 3
        
    def __repr__(self):
        return super().__repr__()    
    
    @doc_inherit
    def _consider_equalities(self):
        return False
    
    @doc_inherit
    def _get_num_colours(self):
        return 3
        
    def __repr__(self):
        return super().__repr__()
   

class HEPGSEnsemble(ImageDescriptor):
    """Combination oh HEPGS descriptors"""
    
    def __init__(self, name, configuration, type_of_features):
        """Default constructor
        
        Parameters
        ----------
        name : str
            Name of the descriptor. For possible values see HEPGS.factory()
        configuration : str
            A string indicating the neighbourhood configuration. Can be:
                'r1p8-r2p8-r3p8' -> Concatenation of tree concetric 
                                    neighbourood of radius 1px, 2px and 3px 
                                    with eight point for each ring.
        feature_type : str
            The type of features to be computed. For possible values see 
            HEPGS.factory()
        """
        super().__init__()
        self._descriptors = list()
        self._name = name
        self._configuration = configuration
        self._type_of_features = type_of_features
        
        if self._configuration == 'r1p8-r2p8-r3p8':
            settings = [{'radius' : 1, 'norm_exp' : 0, 'max_num_points': None},
                        {'radius' : 2, 'norm_exp' : 1, 'max_num_points': 8},
                        {'radius' : 3, 'norm_exp' : 1, 'max_num_points': 8}]            
        else:
            raise Exception('Configuration not supported')
        
        for setting in settings:
            self._descriptors.append(HEPGS.factory(name, 
                                                   setting['radius'],
                                                   setting['norm_exp'],
                                                   setting['max_num_points']))
    
    @doc_inherit        
    def _compute_features(self):
        features = np.array([])
        
        for descriptor in self._descriptors:
            f = descriptor.get_features(self._img_in)
            features = np.concatenate((features, f))
            
        return features
            
    def __repr__(self):
        return '{}-{}-{}'.format(self._name, 
                                 self._configuration, 
                                 self._type_of_features)
                
            