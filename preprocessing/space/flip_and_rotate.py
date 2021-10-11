import numpy as np

from cenotaph.third_parties.doc_inherit import doc_inherit
from cenotaph.basics.base_classes import Image, ImagePreprocessorS

class FlipLr(ImagePreprocessorS):
    """Flip along the vertical axis"""
    
    @doc_inherit    
    def _preprocess(self, img):
        
        #Apply the transformation
        data_out = np.fliplr(m = img.get_data(copy = False))
        
        #Update and return the output image
        img.set_data(data_out)
        return img 
    
class FlipUd(ImagePreprocessorS):
    """Flip along the vertical axis"""
    
    @doc_inherit    
    def _preprocess(self, img):
        
        #Apply the transformation
        data_out = np.flipud(m = img.get_data(copy = False))
        
        #Update and return the output image
        img.set_data(data_out)
        return img    

class UnitRotation(ImagePreprocessorS):
    """Rotation by multiples of 90°"""
    
    def __init__(self, num_rotations):
        """
        Parameters
        ----------
        num_rotations : int
            The number of rotations by 90°. Enter 1 for 90°, 2 for 180, etc.
        """
        self._num_rotations = num_rotations
    
    @doc_inherit    
    def _preprocess(self, img):
        
        #Apply the transformation
        data_out = np.rot90(m = img.get_data(copy = False), 
                            k = self._num_rotations)
        
        #Update and return the output image
        img.set_data(data_out)
        return img