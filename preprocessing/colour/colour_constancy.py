from abc import abstractmethod
from copy import deepcopy

import numpy as np

from cenotaph.preprocessing.colour.base_classes import ColourPreprocessor
from cenotaph.third_parties.doc_inherit import doc_inherit


class ColourConstancy(ColourPreprocessor):
    """Generic base class for colour constancy methods"""
    
    def __init__(self):
        """Default constructor"""
        
        super().__init__()
            
    def _update(self, img_in, new_data, round_and_window=True):
        """A wrap-up function to update the data once the pre-processing is 
        done
        
        Parameters
        ----------
        img_in : Image
            The input image
        new_data : ndarray
            The image data after colour pre-processing
        round_and_window : bool
            Whether the data should be rounded and windowed
            
        Returns
        -------
        None 
            The function modifies the input image
        """

        img_in.set_data(new_data, round_and_window=True)
        
    def get_result(self, img):
        """Get the preprocessed image

        Parameters
        ----------
        img : Image
            The input image. 

        Returns
        ----------
        retval : list of one Image
            The pre-processed image
        """ 
        
        img_out = self._check_input(img)
        
        #Do the colour preprocessing
        data_out = self._preprocess(img_out)
                
        #Update the output and return
        self._update(img_out, data_out, True)
        retval = list()
        retval.append(img_out)
        return retval   
    
    @abstractmethod
    def _preprocess(self, img_in):
        """The colour constancy kernel
        
        Parameters
        ----------
        img_in : Image
            The input image
        
        Returns
        -------
        data_out : ndarray
            The colour output data
        """
      
class Chroma(ColourConstancy):
    """Chromaticity representation [1]. Output returned by get_result is a list
    containing one image.

    References
    ----------
    [1] Cernadas, E., Fernández-Delgado, M., González-Rufino, E., Carrión, P.
        Influence of normalization and color space to color texture classification
        (2017) Pattern Recognition, 61, pp. 120-138
    """
    
    @doc_inherit
    def _preprocess(self, img_in):
        
        #Add one to avoid division by zero
        data_out = img_in.get_data() + 1
        
        #Number of quantisation levels
        G = 2**img_in.get_bit_depth()  
        
        #Set the maximum and minimum theoretical values
        m = 1/(1 + 2*G) #Min
        M = G/(2 + G)   #Max
        
        #Compute normalisation factor (sum of the intensities of the thre channels)
        S = data_out.sum(axis = 2)  
        S = np.repeat(S[:, :, np.newaxis], 3, axis=2)
        
        #Do the normalisation
        data_out = data_out/S
        data_out = (G - 1)*(data_out - m)/(M - m)
                 
        return data_out
        
class GreyWorld(ColourConstancy):
    """Grey world colour normalisation as described in [1]. Output returned by 
    get_result is a list containing one image.

    References
    ----------
    [1] Nikitenko, D., Wirth, M., Trudel, K. 
        Applicability of white-balancing algorithms to restoring faded colour 
        slides: An empirical evaluation
        (2008) Journal of Multimedia, 3 (5), pp. 9-18.
    """ 
    
    @doc_inherit    
    def _preprocess(self, img_in):
        data_in = img_in.get_data()
        data_out = np.empty(data_in.shape)
        
        #Average values by channel
        avgs = data_in.mean(axis=(0,1))
        
        if not all([avgs[0],avgs[2]]) > 0.0:
            raise Exception('Average of red or blue channel is zero')
                
        r_gain = avgs[1]/avgs[0]
        b_gain = avgs[1]/avgs[2]
              
        data_out[:,:,0] = data_in[:,:,0]*r_gain
        data_out[:,:,1] = data_in[:,:,1]
        data_out[:,:,2] = data_in[:,:,2]*b_gain 
        
        return data_out
        
class MaxWhite(ColourConstancy):
    """Max white colour normalisation as desc in [1]*. Output returned by 
    get_result is a list containing one image.

    *Note: in this implementation we replaced 2**bit_depth with 2**(bit_depth-1)
    - See Eq. 4 of [1]
    
    References
    ----------
    [1] Nikitenko, D., Wirth, M., Trudel, K. 
        Applicability of white-balancing algorithms to restoring faded colour 
        slides: An empirical evaluation
        (2008) Journal of Multimedia, 3 (5), pp. 9-18.
    """ 
    
    def _max_white(self, data_in, bit_depth):
        """Colour preprocessing kernel
        
        Parameters
        ----------
        data_in : ndarray (H,W,3)
            The colour input data
        bit_depth : int
            The bit depth
        
        Returns
        -------
        data_out : ndarray (H,W,3)
            The pre-processed data
        """
        
        data_out = np.empty(data_in.shape)
        
        maxs = data_in.max(axis=(0,1))
        
        if not all(maxs) > 0.0:
            raise Exception('Maximum of of at least one of the channels is zero')
        
        gains = 2**(bit_depth - 1)/maxs
        
        data_out[:,:,0] = data_in[:,:,0]*gains[0]
        data_out[:,:,1] = data_in[:,:,1]*gains[1]
        data_out[:,:,2] = data_in[:,:,2]*gains[2]  
        
        return data_out
    
    @doc_inherit       
    def _preprocess(self, img_in):
        return self._max_white(img_in.get_data(), img_in.get_bit_depth())     
        
class Stretch(ColourConstancy):
    """Stretch colour normalisation as implemented in [1]*. Output returned by 
    get_result is a list containing one image.
    
    References
    ----------
    [1] Nikitenko, D., Wirth, M., Trudel, K. 
        Applicability of white-balancing algorithms to restoring faded colour 
        slides: An empirical evaluation
        (2008) Journal of Multimedia, 3 (5), pp. 9-18.
    """ 
    
    @doc_inherit    
    def _preprocess(self, img_in):   
        data_in = img_in.get_data()
        data_out = np.empty(data_in.shape)        
        
        mins = data_in.min(axis=(0,1))
                       
        data_out[:,:,0] = data_in[:,:,0] - mins[0]
        data_out[:,:,1] = data_in[:,:,1] - mins[1]
        data_out[:,:,2] = data_in[:,:,2] - mins[2]
        
        #Now apply MaxWhite
        mw = MaxWhite()
        data_out = mw._max_white(data_out, img_in.get_bit_depth())
        
        return data_out

class None_(ColourConstancy):
    """Null preprocessor - does nothing"""
    
    @doc_inherit
    def _preprocess(self, img_in):   
        data_out = img_in.get_data()                
        return data_out    
    
        
        