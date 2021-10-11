import numpy as np
from skimage.filters import threshold_li, threshold_otsu, threshold_yen

from cenotaph.basics.base_classes import ImagePreprocessor, ImageType

class Thresholding(ImagePreprocessor):
    """Image thresholding"""
    
    #Lookup table for thresholding methods
    _methods_lut = {'Li' : threshold_li, 'Otsu' : threshold_otsu, 
                    'Yen' : threshold_yen}
    
    def __init__(self, method):
        """Default constructor
        
        Parameters
        ----------
        method : str
            A string indicating the thresholding method, Possible values are:
                'Otsu', 'Yen'
        
        References
        ----------
        [2] Sezgin M. and Sankur B. (2004) "Survey over Image Thresholding
            Techniques and Quantitative Performance Evaluation" Journal of
            Electronic Imaging, 13(1): 146-165
            http://www.busim.ee.boun.edu.tr/~sankur/SankurFolder/Threshold_survey.pdf
        """
        try:
            self._thresholding_method = self._methods_lut[method]   
        except KeyError:
            'Thresholding method not supported'
    
    def _threshold(self, data_in):
        """Performs the thresholding
        
        Parameters
        ----------
        data_in : ndarray (height, width, nchannels)
            The input data
            
        Returns
        -------
        data_out : ndarray (height, width, nchannels)
            The thresholded data
        """    
        
        #Iterate through the channels and threshold
        if data_in.ndim == 2:
            t = self._thresholding_method(data_in)
            data_in = data_in <= t
        else:
            for channel in range(data_in.shape[2]):
                t = self._thresholding_method(data_in[:,:,channel]) 
                data_in[:,:,channel] = data_in[:,:,channel] <= t
        
        data_out = data_in.astype(np.uint8)
        return data_out

    def _preprocess(self, img):
        """Generate the preprocessed image

        Parameters
        ----------
        img : Image
            The input image. 

        Returns
        ----------
        img : Image
            The thresholded image. For visualisation reasons this is encoded
            encoded as follows: 
                - values below or equal to threshold = 0 
                - values above threshold = 2^bit_depth - 1 
            where bit_depth is the bit depth of the input image
        """
        
        #Get the results of thresholding (in binary format)
        data_out = self._threshold(img.get_data())
        
        #Set 0 if below or equal to threshold, 2^(num_bits) - 1 otherwise
        #data_out = (2**img.get_bit_depth() - 1) * int(data_out)
        data_out = np.multiply(2**img.get_bit_depth() - 1, data_out)
        
        #Update the internal data in the output image
        img.set_data(data_out)
        
        #Mark the output image as BW
        img.set_image_type(ImageType.BW)

        return img