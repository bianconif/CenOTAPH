from abc import ABC, abstractmethod
from copy import copy, deepcopy
from enum import Enum
from math import floor
from warnings import warn

import numpy as np
import PIL.Image as pimg

import cenotaph.basics.generic_functions as gf

class ImageType(Enum):
    BW = 0          #A binary image
    GS = 1          #A grey-scale image
    RGB = 2         #A three-channel RGB image
    REPEATED = 3    #An n-channel image obtained by repeating a single-channel
                    #one n times
    OTHER = 4        
    
class Image():
    """Wrapper around image data"""
    
    def __init__(self, img_file):
        """Default constructor. Creates an object from an image file.
        
        Parameters
        ----------
        img_file : str
            The image locator - absolute or relative path
        """
        
        #Store the source file
        self._img_file = img_file
        
        #Open the input image and check the format
        _img = pimg.open(self._img_file)
        
        #Store the image data as ndarray
        self._data = np.asarray(_img)         
        
        #Determine and store the image type
        self._img_type = ImageType.OTHER
        if (_img.mode == 'RGB') or (_img.mode == 'RGBA'):
            self._img_type = ImageType.RGB
        elif _img.mode == 'L':
            self._img_type = ImageType.GS
        else:
            raise Exception('Image type not supported')
        
        self._set_bit_depth()
        self._set_n_channels()
    
    @classmethod
    def from_array(cls, data, img_type):
        """Create an object from a numpy array
        
        Parameters
        ----------
        data : nparray (H,W) or (H,W,C)
            The input data, where H, W and C are the height, width and number 
            of channels. Must be a numpy array of dtype = uint8, uint16,
            uint32 or uint 64. 
        img_type : ImageType
            The image type - i.e., RGB, GS or BW
        
        Returns
        -------
        img : Image
            The Image object.
        """
        img = cls.__new__(cls)
        img._data = np.copy(data)
        img._img_file = None
        img._img_type = img_type
        img._set_bit_depth()
        img._set_n_channels()     
        return img
        
    
    def copy(self):
        """Returns a deepcopy"""
        return(deepcopy(self))
            
    def _set_n_channels(self):
        """Determine and store the number of channels"""
        self._num_channels = None
        _dims = len(self._data.shape)
        if _dims == 3:
            self._num_channels = self._data.shape[2]
        elif _dims == 2:
            self._num_channels = 1
        else:
            raise Exception('Dimension of the input image unsupported')        
    
    def get_data_type(self):
        """Get the inner data type
        
        Returns
        -------
        data_type : np.dtype
            The type of the inner data.
        """
        _probe = (np.ravel(self._data))[0]
        return type(_probe)
    
    def _set_bit_depth(self):
        """Determine and store the bit depth"""
        
        _probe = (np.ravel(self._data))[0]
        self._bit_depth = None
        if isinstance(_probe, np.uint8):
            self._bit_depth = 8
        elif isinstance(_probe, np.uint16): 
            self._bit_depth = 16
        elif isinstance(_probe, np.uint32):
            self._bit_depth = 32
        elif isinstance(_probe, np.uint64):
            self._bit_depth = 64
        else:
            raise Exception('Bit depth not supported')        
        
    def get_size(self):
        """The image size"""
        return self._data.shape
        
    def get_n_channels(self):
        """Get the number of channels
        
        Returns
        -------
        The number of channels (int)
        """
        
        return self._num_channels
    
    def get_bit_depth(self):
        """Get the bit depth
        
        Returns
        -------
        The bit depth per channel (int)
        """
        
        return self._bit_depth  
    
    def normalised_as_float(self):
        """Return the inner data as float normalised in the [0,1] interval
        
        Returns
        -------
        normalised_data : nparray of float
            The normalised data.
        """
        return self.get_data().astype('float')/(2**self.get_bit_depth() - 1)
    
    def get_one_hot(self, lut):
        """Get one-hot encoding from single-channel images
        
        Parameters
        ----------
        lut : list of numerics (N)
            The lookup-table (list of values to search in the image).
        
        Returns
        -------
        one_hot_encoding : nparray of int (H,W,N)
            A stack of one-hot layers, one for each of the entries inthe lut
        """
        
        if self.get_n_channels() > 1:
            raise Exception('The input image needs to be single channel')
        
        N = len(lut)
        H, W = self.get_size()[0:2]
        one_hot_encoding = np.zeros((H,W,N)).astype(np.uint8)
        for n, label in enumerate(lut):
            one_hot_encoding[:,:,n] = (self._data == label)
            
        return one_hot_encoding
    
    def get_data(self, copy=True):
        """Return the image data
        
        Parameters
        ----------
        copy : bool
            If True the function returns a copy of the data, otherwise
            a reference to them. Use True when you need the original data not
            to be modified
        """
        
        data = None
        if copy:
            data = self._data.copy()
        else:
            data = self._data
        
        return data
    
    def get_type(self):
        """Get the image type"""
        
        return self._img_type
    
    def set_data(self, data):
        """Set the image data
        
        Parameters
        ----------
        data : ndarray of uint
            The image data (pixel values)
        """
        
        #Copy the input data
        data_copy = np.copy(data)
        
        #Update the data values. The inner representation (np.dtype) does not
        #change
        self._data = data_copy.astype(self.get_data_type())
        
        #Update bit depth and number of channels
        self._set_bit_depth()
        self._set_n_channels()
        
    def set_image_type(self, img_type):
        """Set the image type
        
        Parameters
        ----------
        img_type : ImageType
            The image type
        """
        
        self._img_type = img_type
    
    def get_multichannel(self, n=3):
        """From a single-channel image get a multi-channel one by copying
        the original channel n times.
        
        Parameters
        ----------
        n : int
            The number of channels
            
        Returns
        -------
        img_out : Image
            The multi-channel image.
        """
        
        if not self.get_n_channels() == 1:
            raise Exception('The input image needs to be single-channel')
        
        data_in = self.get_data(copy = True)
        data_in = np.expand_dims(data_in, axis = -1)
        data_out = np.repeat(data_in, n, axis = 2)
        img_out = self.from_array(data_out, ImageType.REPEATED)
        return img_out
    
    def to_greyscale(self, conversion='Luminance'):
        """RGB to greyscale conversion
        
        Parameters
        ----------
        conversion : str
            The type of grey-scale conversion. Can be:
                'Intensity' -> mean of the RGB channels
                'Luminance' -> weighted mean of the RGB channels
            See [1] for details.
                
        References
        [1] Kanan, C., Cottrell, G.W.
            Color-to-grayscale: Does the method matter in image recognition?
            (2012) PLoS ONE, 7 (1), art. no. e29740, .
        """
        
        #Can't convert if not RGB
        if self._img_type is not ImageType.RGB:
            raise Exception('Image is not RGB, cannot convert to grey-scale')
        
        #No need to convert if already GS
        if self._img_type is not ImageType.GS:
            
            r, g, b = self._data[:,:,0], self._data[:,:,1], self._data[:,:,2]
        
            if conversion == 'Intensity':
                gs = (1 / 3) * (r + g + b)
            elif conversion == 'Luminance':
                gs = 0.3 * r + 0.59 * g + 0.11 * b
            else:
                raise Exception('Conversion type *' + conversion + '* not supported')
        
            self._data = np.around(gs) 
            
            #Cast back to int
            self._cast_data_type_by_bit_depth()
            
            #Update bit depth and number of channels
            self._set_bit_depth()
            self._set_n_channels()    
            
            #Update image type
            self._img_type = ImageType.GS

    def invert(self):
        """Invert colours"""
        
        pivot = floor(2**(self.get_bit_depth() - 1) - 0.5)
        original = self.get_data()
        inverted = -(original - pivot)
        self.set_data(inverted)

    def _cast_data_type_by_bit_depth(self):
        """Set the appropriate data type for the bit depth of the image"""
        
        if self._bit_depth == 8:
            self._data = (np.uint8)(self._data)
        elif self._bit_depth == 16: 
            self._data = (np.uint16)(self._data)
        elif self._bit_depth == 32: 
            self._data = (np.uint32)(self._data)
        elif self._bit_depth == 64: 
            self._data = (np.uint64)(self._data)
        else:
            raise Exception('Bit depth not supported')         

    def _round_and_window(self):
        """Window the data in the range [0,2^bit_depth - 1] and round to int 

        Parameters
        ----------
        none
        """   
        
        #Round
        _new_data = self._data.round()
        
        #Window
        _new_data[_new_data < 0.0] = 0
        _new_data[_new_data > 2**self._bit_depth - 1] = 2**self._bit_depth - 1        
        
        #Update the data
        self._data = _new_data

    def get_num_pixels(self):
        """Number of pixels in the image
        
        Returns
        -------
        num_pixels : int
            The number of pixels in the image
        """
        
        num_points = self._data.shape[0] * self._data.shape[1]
        return num_points
    

    def print_info(self):
        """Print image info"""
        
        print('Source file: ' + self._img_file)
        print('Image type: ' + str(self._img_type))
        print('Bit depth: ' + str(self.get_bit_depth()))
        print('Number of channels: ' + str(self.get_n_channels()))
        
    def save(self, destination):
        """Save image data to disk
        
        Parameters
        ----------
            dest : full (relative or absolute) path (including extension) 
            of the destination file
        """
        im = pimg.fromarray(self._data)
        im.save(destination)  
        
    def get_bwimage(self):
        """Returns the binary format of the image, i.e. 0 where the original
        value is 0, 1 otherwise. Designed to work with images of type
        ImageType.BW. The original image is not modified.
        
        Returns
        -------
        bwimage : nparray of int, same size as the image
            The image in binary (0,1) format.
        """
        
        #Issue a warning if the function is called on a non-BW image
        if self.get_type() is not ImageType.BW:
            warn('get_bwimage() called on a non-BW image')
        
        #Get a copy of the data
        bwimage = self.get_data(copy = True)
        
        #Get the original data type
        data_type = self.get_data_type()
        
        #Convert to binary
        bwimage = (bwimage > 0).astype(data_type)
        
        return bwimage
        
class ImageHandler(ABC):
    """Generic image processor, superclasses ImageDescriptor and ImagePreprocessor""" 
    
    def __init__(self):
        """Default constructor"""
            
        #Initialise the input image as None
        self._img_in = None 
        
    def set_input_image(self, img):
        """Make a deep copy of the input image and store it as a member
        variable
        """
        #Make a deep copy of the input image to be sure the original image is
        #not modified
        self._img_in = deepcopy(img)  
                                   
class ImageDescriptor(ImageHandler):
    """Generic image descriptor (abstract base class)"""

    def __init__(self):
        """Default constructor"""
        
        #Invoke the superclass constructor
        super().__init__()
                                
    def __repr__(self):
        return self.__class__.__name__

    @abstractmethod
    def _compute_features(self):
        """Compute the features. Abstract method, needs to be be implemented in the
        subclasses

        Parameters
        ----------
        none

        Returns
        -------
        features : ndarray (N)
            The features
        """

    def get_features(self, img, normalisation=None):
        """Get the features

        Parameters
        ----------
        img : Image
            The input image. 
        normalisation : int
            The exponent of the Minkowsky distance used to normalise the 
            features. Pass None if no normalisation is required.
            
        Returns
        ----------
        fv : ndarray
            The feature vector
        """
        fv = None
        
        #Store the input image and compute the features
        self.set_input_image(img)   
                
        #Compute and return the features
        fv = self._compute_features()
        
        #Normalise the features if required
        if normalisation is not None:
            fv_norm = np.linalg.norm(fv, ord=normalisation)
            fv = fv/fv_norm
        
        return fv

class ImageDescriptorGS(ImageDescriptor):
    """Generic grey-scale image descriptor"""
    
    def __init__(self, gsconversion='Luminance'):
        """Default constructor
        
        Parameters
        ----------
        gsconversion: str
            The algorithm for greyscale conversion if the input is RGB. 
            For possible values see basics.rgb2gray()
        """

        super().__init__()
        self._gsconversion = gsconversion

    def set_input_image(self, img):
        """Reads the input image, converts it into greyscale if required 
        and stores it as a member variable
        
        Parameters
        ----------
        img : Image
            The input image.  
        """
                
        super().set_input_image(img)
        
        #Convert to greyscale if required
        if not self._img_in.get_type() == ImageType.GS:
            self._img_in.to_greyscale(self._gsconversion)
        
            
    #def standardise(self):
        #"""Zero mean and unit variance standardization
        
        #Parameters
        #----------
            #bit_depth : int
                #The number of bits used for image encoding
        #"""
        
        ##Normalize into the [0,1] interval
        #nimg = self._img/(2**bit_depth)
        
        ##Normalize to zero mean and unit variance
        #mean = np.mean(nimg)
        #var = np.var(nimg)
        #nimg = (nimg - mean)/var
        
class ImagePreprocessorS(ImageHandler):
    """Abstract base class for image preprocessing when the preprocessing
    operation returns only one image for each input image""" 
    
    def __repr__(self):
        return self.__class__.__name__    
    
    @abstractmethod 
    def _preprocess(self, img):
        """Abstract method to be implemented by the subclasses.
        
        Parameters
        ----------
        img : Image
            The input image. 

        Returns
        -------
        img_out : Image
            The pre-processed image
        """    
    
    def get_result(self, img):
        """Get the preprocessed images. The input image is not modified.

        Parameters
        ----------
        img : Image
            The input image. 

        Returns
        ----------
        img_out : Image
            The pre-processed image
        """
        
        #Copy the original image
        img_copy = img.copy()
        
        #Preprocess the copy and return the result
        img_out = self._preprocess(img_copy)        
        return img_out    

class ImagePreprocessorM(ImageHandler):
    """Abstract base class for image preprocessing when the preprocessing
    operation returns more than one image for each input image"""
                
    def __repr__(self):
        return self.__class__.__name__
    
    @abstractmethod 
    def _preprocess(self, img):
        """Abstract method to be implemented by the subclasses.
        
        Parameters
        ----------
        img : Image
            The input image. 

        Returns
        -------
        imgs_out : list of Image
            The pre-processed images
        """        
                        
    def get_result(self, img):
        """Get the preprocessed images. The input image is not modified.

        Parameters
        ----------
        img : Image
            The input image. 

        Returns
        ----------
        imgs_out : list of Image
            The pre-processed images
        """
        
        #Copy the original image
        img_copy = img.copy
        
        #Preprocess the copy and return the result
        imgs_out = self._preprocess(img_copy)        
        return imgs_out
       
class ImageProcessingPipeline():
    """Top-level container class for defining a complete image processing 
    pipeline including pre-processing, feature extraction and post-processing""" 
    
    def __init__(self, image_descriptors, pre_processors=None, verbose=True):
        """Default constructor
        
        Parameters
        ----------
        image_descriptors : list of ImageDescriptor instances
            The image decriptor(s) used.
        pre_processors : list of ImagePreprocessor instances
            The image pre-processor(s) used. 
        verbose : bool
            Print messages as it goes
        
        NOTE: the feature vectors are computed for each of the combinations
        pre_processor x image_descriptors, and the corresponding results
        concatenated.
        """
        
        self._pre_processors = pre_processors
        self._image_descriptors = image_descriptors
        self._verbose = verbose
         
    def compute_features(self, img):
        """Feature extraction
        
        Parameters
        ----------
        img : Image
            The input image. 
            
        Returns
        ----------
        features : ndarray of float
             The image features 
        """       
        
        if self._verbose:
            print('Computing features: ' + img._img_file)
        
        #Initialise the output
        features = np.array([])
                
        #Do the pre-processing
        _pre_processed_images = list()
        if not self._pre_processors:
            #If no pre-processor is given use the input image as is
            _pre_processed_images.append(img)
        else:
            for preproc in self._pre_processors:
                #Get the results of this pre-processor
                res = preproc.get_result(img)
                
                #Unpack the results and append them to _pre_processed_images
                for r in res:
                    _pre_processed_images.append(r)
                    
        #Do the feature calculation
        for prepimg in _pre_processed_images:
            for imdesc in self._image_descriptors:
                features_ = imdesc.get_features(prepimg)
                try:
                    features = np.concatenate((features, features_))
                except ValueError:
                    features_ = np.expand_dims(features_, axis = -1)
                    features = np.concatenate((features, features_))
                    
            
        
        return features    
    
    @classmethod        
    def save_features(cls, features, out):
        """Saves the features on a file
        
        Parameters
        ----------
        features : ndarray
            The features to be saved
        out : str
            Full or relative path to the output file
        """
        
        try:
            np.save(out, features)
        except:
            raise Exception('Something went wrong when trying to save features')
        
    @classmethod
    def load_features(cls, source):
        """Reads the features from a file
        
        Parameters
        ----------
        source : str
            Full or relative path to the file where the features are stored
        
        Returns
        -------
        features : ndarray
            The features
        """ 
        
        features = None
        try:
            features = np.load(source)
        except:
            raise Exception('Something went wrong when trying to load features')   
        return features
                
class Ensemble():
    """Ensemble of image descriptors"""
    
    def __init__(self, image_descriptors, combination_mode = 'concat', 
                     normalisation_mode = 'by-descriptor', order=1):
        """
        
        Parameters
        ----------
        image_descriptors: list of ImageDescriptor
            The image descriptors to combine in the ensemble.
                combination_mode: str
            How to combine the feature vectors generated by each descriptor.
            Can be:
                'concat' -> The feature vectors are concatenated
        normalisation_mode: str  
            How to normalise the features. No normalisation is applied if 
            this parameter is not provided . Possible values are:
                'global'        -> The normalisation is applied to the final 
                                   feature vector
                'by-descriptor' -> The normalisation is applied to each feature
                                   vector separately and before they are 
                                   combined into the final vector.
        order: int
            Order of the norm for feature normalisation. Default = 1.
        """
        self._descriptors = image_descriptors
        self._combination_mode = combination_mode
        self._normalisation_mode = normalisation_mode
        self._order = order
        
    def get_features(self, img):
        """
        Parameters
        ----------
        img : Image
            The input image. 
            
        Returns
        ----------
        fv : ndarray
            The feature vector
        """  
        
        fv = np.empty([0])
                
        if self._combination_mode == 'concat':
            if self._normalisation_mode == 'global':
                for descriptor in self._descriptors:
                    fv = np.hstack((fv, descriptor.get_features(img)))
                fv = np.linalg.norm(fv, ord=order)
            elif self._normalisation_mode == 'by-descriptor':
                for descriptor in self._descriptors:
                    fv_ = descriptor.get_features(img, normalisation = self._order)
                    fv = np.hstack((fv, fv_)) 
                    a = 0
            else:
                raise Exception(f'Normalisation mode {normalisation_mode} not '
                                f'supported')
        else:
            raise Exception(f'Combination mode {combination_mode} not supported')
        
        return fv
        
        
        
        
       
        
    