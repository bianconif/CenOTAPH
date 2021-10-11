from copy import deepcopy

import numpy as np
import PIL.Image as PImage

from cenotaph.third_parties.doc_inherit import doc_inherit           
from cenotaph.basics.base_classes import ImagePreprocessorM,\
     ImagePreprocessorS

class Crop(ImagePreprocessorS):
    """Image crop"""
    
    def __init__(self, offset, new_size):
        """Default (empty) constructor
        
        Parameters
        ----------
        offset : tuple of int (v_offset, h_offset)
            The vertical and horizontal offset from the top left corner
        new_size : tuple of int (new_height, new_width)
            The height and width of the cropped image
        """
        
        self._offset = offset
        self._new_size = new_size
    
    def _crop(self, data_in, offset, new_size):
        """Performs the crop
        
        Parameters
        ----------
        data_in : ndarray (height, width, nchannels)
            The input data
        offset : tuple of int (v_offset, h_offset)
            The vertical and horizontal offset from the top left corner
        new_size : tuple of int (new_height, new_width)
            The height and width of the cropped image
            
        Returns
        -------
        data_out : ndarray (new_height, new_width, nchannels)
            The cropped data
        """
        
        #Check the input values
        if (len(offset) != 2) or (len(new_size) != 2):
            raise Exception(\
                'Parameters offset and new_size should have length two')
        if (offset[0] < 0.0) or (offset[1] < 0.0):
            raise Exception('Offset cannot be negative')
        if (new_size[0] < 0.0) or (new_size[1] < 0.0):
            raise Exception('Crop size cannot be negative') 
        if (new_size[0] > data_in.shape[0]) or\
           (new_size[1] > data_in.shape[1]):
            raise Exception('Crop area cannot be larger than the input image')
                
        #Do the slicing
        try:
            data_out = data_in[offset[0]:offset[0]+new_size[0],
                               offset[1]:offset[1]+new_size[1],:]
        except IndexError:
            data_out = data_in[offset[0]:offset[0]+new_size[0],
                               offset[1]:offset[1]+new_size[1]]            
        
        return data_out

    @doc_inherit
    def _preprocess(self, img):        
        img_out = self._crop(img.get_data(), self._offset, self._new_size)
        return img_out

class CentralCrop(ImagePreprocessorS):
    """Crop image from the center"""
    
    def __init__(self, new_size):
        """Default constructor
        
        Parameters
        ----------
        new_size : tuple of int (height, width) 
            The height and width of the cropped image
        """
        
        self._new_size = new_size
    
    @doc_inherit
    def _preprocess(self, img):

        shape = img.get_data().shape
        
        #Check the input values
        if (self._new_size[0] > shape[0]) or (self._new_size[1] > shape[1]):
            raise Exception('Crop area cannot be larger than the input image')
        
        #Determine the offsets
        v_offset = (shape[0] - self._new_size[0]) // 2
        h_offset = (shape[1] - self._new_size[1]) // 2
        
        #Perform the crop and return the result
        cropper = Crop((v_offset,h_offset),self._new_size)
        return cropper.get_result(img)

class Resize(ImagePreprocessorS):
    """Image resizing"""
    
    def __init__(self, new_size, interp='bicubic'):
        """The constructor
        
        Parameters
        ----------
        new_size : int (height, width)
            A tuple or list conatining the dimension after resizing.
        interp : str
            The interpolation algorithm to use for re-sizing. Can be: 
                'nearest'  -> Nearest neighbour interpolation
                'lanczos'  -> Lanczos interpolation 
                'bilinear' -> Bilinear interpolation 
                'bicubic'  -> Bicubic interpolation

        """
        super().__init__()
        
        self._new_size = new_size
        
        if interp == 'nearest':
            self._interp = PImage.NEAREST
        elif interp == 'bilinear':
            self._interp = PImage.BILINEAR
        elif interp == 'bicubic':
            self._interp = PImage.BICUBIC
        elif interp == 'lanczos':
            self._interp = PImage.LANCZOS
        else:
            raise Exception('Interpolation method not supported')
    
    @doc_inherit
    def _preprocess(self, img):
        data_in = img.get_data()
        data_out = np.array(PImage.fromarray(data_in).resize(self._new_size,
                                                             self._interp))
        img.set_data(data_out)
        return img
    
class UniformSplit(ImagePreprocessorM):
    """Partitions the input image into a set of equally-sized sub-images"""
    
    def __init__(self, num_splits):
        """The default constructor
        
        Parameters
        ----------
        num_splits : tuple of int (v_splits, h_splits)
            Number of vertical (v_splits) and horizontal (h_splits) 
            subdivisions.
        """
        
        self._num_splits = num_splits
    
    @doc_inherit    
    def _preprocess(self, img):
         
        height, width = img.get_data().shape[0:2]
        n_channels = img.get_n_channels()
        
        #Compute the dimension of the tiles
        tile_width = width // self._num_splits[1]
        tile_height = height // self._num_splits[0]
        
        #Do a central crop to match the tiles' size
        new_height = tile_height * self._num_splits[0]
        new_width = tile_width * self._num_splits[1]        
        if (height, width) != (new_height, new_width):
            cropper = CentralCrop((new_height, new_width))
            img = cropper.get_result(img)[0]
        
        #Do the slicing
        retval = list()    
        for h in range(self._num_splits[0]):
            for w in range(self._num_splits[1]):
                offset = (h * tile_height, w * tile_width)
                cropper = Crop(offset, (tile_height, tile_width))
                retval.append(cropper.get_result(img)[0])
                
        return retval