"""Contrast and gamma correction"""
import numpy as np
from skimage.exposure import adjust_gamma

from cenotaph.third_parties.doc_inherit import doc_inherit
from cenotaph.basics.base_classes import Image, ImagePreprocessorS

class Gamma(ImagePreprocessorS):
    """Gamma correction"""
    
    def __init__(self, gamma = 1.0):
        """
        Parameters
        ----------
        gamma : float
            Exponent of gamma correction. Needs to be non-negative.
        """
        self._gamma = gamma
        
    @doc_inherit    
    def _preprocess(self, img):
        
        #Get the original data type
        data_type = img.get_data_type()
        
        #Apply the transformation
        data_out = adjust_gamma(image = img.get_data(copy = False),
                                gamma = self._gamma,
                                gain = 1.0).astype(dtype = data_type)
        
        #Update and return the output image
        img.set_data(data_out)
        return img     