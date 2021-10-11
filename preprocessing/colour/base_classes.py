from copy import deepcopy

from cenotaph.basics.base_classes import ImagePreprocessor

class ColourPreprocessor(ImagePreprocessor):
    """Generic colour preprocessor (abstract base class)"""

    def __init__(self):
        """Default constructor"""        
        
        #Invoke the baseclass constructor
        super().__init__()
        
    def _check_input(self, img):
        """Make sure the input image is three-channel and return a deep copy
        of it
        
        Parameters
        ----------
        img : Image
            The input image. 
        Returns
        -------
        img_out : Image
            A deep copy of the input image
        """          
        
        #Check the image is a three-channel one
        if not (img.get_n_channels() == 3):
            raise Exception('The input image needs to be three-channel') 
        
        return deepcopy(img)
        
        