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
        
class LBPBasics(HEP):
    """Basic functions for LBP-like descriptors"""
    
    @staticmethod
    def _consider_equalities():
        return False
    
    @staticmethod
    def _get_num_colours():
        return 2  
    
    @staticmethod
    def _get_thresholds():
        return [0]
    
    def _get_dictionary(self):
        dictionary = list(range(self._get_num_colours() **\
                          self._get_num_peripheral_points()))
        return dictionary    
        
class LBP(LBPBasics, HEPGS, HEPLocalThresholding):
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
        return [self._gs_layers[:,:,self._neighbourhood.center_index()]]

    @doc_inherit
    def _get_base_values(self):
        return [self._gs_layers[:,:,self._neighbourhood.peripheral_indices()]]
                         
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
                
            