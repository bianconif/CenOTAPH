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

class HEPGS(HEP):
    
    def _pre_compute_features(self):
        
        #Convert the input image to greyscale if required
        self._img_in.to_greyscale(conversion='Luminance')
        
        #Generate displaced copies of the input image
        self._gs_layers = matrix_displaced_copies(self._img_in.get_data(), 
                                                  self._neighbourhood.\
                                                  get_integer_points())            

class HEPLocalThresholding(HEP):
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
        
    def _get_pattern_maps(self):
        
        self._pre_compute_features()
                
        patterns = self._generate_patterns_by_thresholding(
            self._get_base_values(), self._get_pivot(), 
            self._get_thresholds(), self._consider_equalities(),
            self._get_weights())  
        
        return [patterns]
         
        
class LBP(HEPGS, HEPLocalThresholding):
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
        return self._gs_layers[:,:,self._neighbourhood.center_index()]

    @doc_inherit
    def _get_base_values(self):
        return self._gs_layers[:,:,self._neighbourhood.peripheral_indices()]
                        
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
                
            