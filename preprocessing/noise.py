import numpy as np

from cenotaph.third_parties.doc_inherit import doc_inherit
from cenotaph.basics.base_classes import ImagePreprocessorS

class Gaussian(ImagePreprocessorS):
    """Additive Gaussian noise"""
    
    def __init__(self, mean=0.0, sigma=0.05):
        """
        Parameters
        ----------
        mean : float
            The central (expected) value of the noise.
        sigma : float
            The standard deviation of the noise.
        
        NOTE: The noise will be scaled accordingly to the bit depth (n) of the
        image to which it is appplied. Therefore, given 'e' the random value
        drawn from a normal distribution with avg = mean and std = sigma, the 
        noise applied to the input image will be E = e*(2**n - 1).
        """
        self._mean = mean
        self._sigma = sigma
    
    def _preprocess(self, img):
        
        #Get the data type of the original image
        data_type = img.get_data_type()
        
        #Get the bit depth
        n = img.get_bit_depth()
        
        #Generate a random array the same size as the input data
        data_in = img.get_data(copy = False)
        rand_noise = np.random.normal(loc = self._mean, 
                                      scale = self._sigma, 
                                      size = data_in.shape)
        
        #Scale the noise value to the bit depth of the image
        rand_noise = np.multiply(rand_noise, 2**n - 1)
        
        #Add the random noise
        data_out = data_in + rand_noise
        
        #Threshold the values outside the range
        data_out[np.where(data_out < 0.0)] = 0.0
        data_out[np.where(data_out > (2**n - 1))] = (2**n - 1)
        
        #Cast the data back to the original type
        data_out = data_out.astype(dtype = data_type)
        
        #Update and return the output image
        img.set_data(data_out)
        return img   
        
        