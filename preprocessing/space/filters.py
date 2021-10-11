from abc import abstractmethod
import numpy as np
from skimage.filters import gaussian

from cenotaph.third_parties.doc_inherit import doc_inherit
from cenotaph.basics.base_classes import Image, ImagePreprocessorS

class Gaussian(ImagePreprocessorS):
    """Gaussian filter"""
    
    def __init__(self, sigma = 1.0):
        """
        Parameters
        ----------
        sigma : float
            The amplitude (standard deviation) of the filter
        """
        self._sigma = sigma
    
    @doc_inherit    
    def _preprocess(self, img):
        
        #Get the data type of the original image
        data_type = img.get_data_type()
        
        #Work the data
        data_out = gaussian(img.get_data(copy = False), 
                            sigma = self._sigma, 
                            output = None, 
                            mode = 'wrap', 
                            cval = 0, 
                            multichannel = True, 
                            preserve_range = True, 
                            truncate = 3.0) 
        
        #Cast the data back to the original type
        data_out = data_out.astype(dtype = data_type)
        
        #Update and return the output image
        img.set_data(data_out)
        return img      
    

    