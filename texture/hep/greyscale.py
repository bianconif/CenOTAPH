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

class LBPDict(HEP):
    def _get_weights(self):
        weights = self._get_num_colours() **\
            np.arange(self._get_num_peripheral_points())
        return weights
    
    def _get_dictionary(self):
        dictionary = list(range(self._get_num_colours() **\
                          self._get_num_peripheral_points()))
        return dictionary
      
    def _compute_invariant_dictionary(self):
        retval = group_invariant_dictionary(
            self._get_dictionary(),
            num_colours = self._get_num_colours(),
            group_action = self._get_group_action()) 
        return retval   
    
class ILBPDict(HEP):
    def _get_weights(self):
        weights = self._get_num_colours() **\
            np.arange(self._get_num_points())
        return weights 
    
    def _get_dictionary(self):
        dictionary = list(range(self._get_num_colours() **\
                          self._get_num_points()))
        return dictionary  
    
    def _compute_invariant_dictionary(self):
        center_index = self._get_center_index()
        retval = group_invariant_dictionary(
            self._get_dictionary(),
            num_colours = self._get_num_colours(),
            group_action = self._get_group_action(),
            excluded_point = center_index) 
        return retval     
    
        
class LBPBasics():
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
            
class LBP(LBPBasics, LBPDict, HEPGS, HEPLocalThresholding):
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
    
class ILBP(LBPBasics, ILBPDict, HEPGS, HEPLocalThresholding):
    """Improved Local binary patterns
    
    References
    ----------
    [1] Jin, H., Liu, Q., Lu, H., Tong, X.
        Face detection using improved LBP under bayesian framework
        (2004) Proceedings - Third International Conference on Image and 
        Graphics, pp. 306-309
    """
        
    def _get_weights(self):
        weights = self._get_num_colours() **\
            np.arange(self._get_num_points())
        return weights
    
    def _get_dictionary(self):
        dictionary = list(range(self._get_num_colours() **\
                          self._get_num_points()))
        return dictionary         
        
    @doc_inherit
    def _get_pivot(self):
        return [np.mean(self._gs_layers, axis=2)]

    @doc_inherit
    def _get_base_values(self):
        return [self._gs_layers]
                         
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
                
            