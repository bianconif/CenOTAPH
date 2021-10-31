import math

import scipy.signal as sig
import numpy as np
import tensorflow as tf
from tensorflow.nn import conv2d
from tensorflow.python.eager.context import eager_mode

import cenotaph.basics.base_classes as bc
from cenotaph.third_parties.doc_inherit import doc_inherit

class ImageFiltering(bc.SingleChannelImageDescriptor):
    """Generic image image filtering. Implementation based on TensorFlow"""
    
    def __init__(self, filter_bank, feature_type = 'MV', \
                 mode='SAME', stride=1, gsconversion='Luminance'):
        """Constructor    
        
        Parameters
        ----------
        filter_bank : ndarray
            The filters to convolve the input image with. A (W,W,N) matrix
            where N is the number of filters
        feature_type : str
            The global features to extract from each transformed image. Can be:
                'M'  ->  Mean of each transformed image
                'MV' ->  Mean and standard deviation of each transformed image
            The features returned are sorted by type first, then by filter in the
            filter bank. For example, given a bank of F filters 
            {f1,...fj,...fF} and feature_type = 'MV' the feature vector will
            be arranged as follows: {M1,...Mj,...,MF,V1,...,Vj,...,VF}
        mode : str
            A string indicating the size of the output. Can be:
                'SAME'  -> transformed images have the same size as the input
                           images
                'VALID' -> transormed images have the same size of the input
                           images minus the size of the filter 
        stride : int or list of ints of  length 1 or 2
            The stride of the sliding window for each dimension of input. 
            If a single value is given it is replicated in the H and W 
            dimension.
        gsconversion : str
            The algorithm for grey-scale conversion if the input is RGB.
            For possible values see Image.to_greyscale().
        """
        
        super().__init__()
        
        self._filter_bank = filter_bank
        self._feature_type = feature_type
        self._mode = mode
        self._stride = stride
        self._gsconversion = gsconversion
        
        #Set the filter bank
        self.set_filter_bank(filter_bank)
        
        #Set the flag for filtering done
        self._filtering_done = False
    
    #doc_inherit
    def set_input_image(self, img):
        super().set_input_image(img)
        if img is not self._img_in:
            self._filtering_done = False
                           
    def set_filter_bank(self, filter_bank):
        """Store the filter bank"""
        
        #Check the shape and dimension of the filter bank
        if len(self._filter_bank.shape) == 2:
            self._num_layers = 1
        elif len(self._filter_bank.shape) == 3:
            self._num_layers = self._filter_bank.shape[2]
        else:
            raise Exception('Dimension of the filter bank is incorrect')
                   
        #If one layer promote to a three-dimensional matrix
        #with one layer
        if self._num_layers == 1:
            self._filter_bank = np.expand_dims(self._filter_bank, 2)
        
        #Make the shape of the filter bank compatible with TensorFlow
        self._filter_bank = np.expand_dims(self._filter_bank, -2)
        
        #Store the data type    
        self._data_type = type(self._filter_bank.flatten()[0])
         
        #Flag to indicate whether the transformed images have been computed
        self._filtering_done = False        
    
    def _compute_features(self):
        """Implements the abstract base class virtual method"""
        
        features = np.array([])
        
        #Perform the filtering
        if not self._filtering_done:
            self._do_filtering()
            
        #Extract the features
        if self._feature_type == 'M':
            M = np.mean(self._transformed_images, (0,1))
            features = M
        elif self._feature_type == 'MV':
            M = np.mean(self._transformed_images, (0,1))
            V = np.std(self._transformed_images, (0,1))
            features = np.concatenate((M,V))
        else:
            raise Exception('Feature type not supported')
        
        return features
          
    def get_filter_bank(self):
        """The filetr bank
        
        Returns
        -------
        self._filter_bank : ndarray (H,W,N)
            The filter bank, where H = image height, W = image width and 
            N = number of filters.
        """
        
        return np.squeeze(self._filter_bank)
        
            
    def get_transformed_images(self, img):
        """Return the transformed images
        
        Parameters
        ----------
        img : Image
            The input image
        """
        
        self.set_input_image(img)
        
        if img is not self._img_in:
            self._filtering_done = False
        
        if not self._filtering_done:
            self._do_filtering()
            
        return self._transformed_images
    
    
    def _do_filtering(self):
        """Perform the filtering"""
        
        #Prepare data for filtering
        to_filter = np.array(self._img_in.get_data()).astype(self._data_type)
        to_filter = np.expand_dims(to_filter, 0)
        to_filter = np.expand_dims(to_filter, -1)

        #Do the filtering
        with eager_mode():
            #Define a context to avoid clashes eager/graph mode in TensorFlow
            assert tf.executing_eagerly()
            res = conv2d(to_filter,
                         self._filter_bank,
                         strides = self._stride,
                         padding = self._mode,
                         data_format='NHWC',
                         name=None)

        
        self._transformed_images = np.squeeze(res.numpy())
        
        ##Apply the first filter
        #_fltr = self._filter_bank[:,:,0] 
        #self._transformed_images = sig.fftconvolve(self._img_in.get_data(), _fltr, \
                                                   #self._mode)        
        
        ##Promote the result to a three-dimensional matrix
        #self._transformed_images = np.expand_dims(self._transformed_images, 2)
        
        ##Apply the other filters and pile up the results
        #for i in np.arange(1,self._num_layers):
            #_fltr = self._filter_bank[:,:,i]
            #_transformed_image = sig.fftconvolve(self._img_in.get_data(), _fltr, \
                                                 #self._mode)            
            #_transformed_image = np.expand_dims(_transformed_image, 2)
            
            #self._transformed_images = np.concatenate((self._transformed_images, \
                                                       #_transformed_image), 2)
                                
        #Update the flag    
        self._filtering_done = True
         
        
class DCF(ImageFiltering):
    """Image features based on Discrete Cosine Filters"""
    
    def __init__(self, feature_type='MV', mode='SAME', stride=1, 
                 gsconversion='Luminance', filter_size=5, num_freqs=5):
        """The constructor
        
        Parameters
        ----------
        See ImageFiltering.__init__() for the meaning of feature_type, mode, 
            stride and gsconversione
        See dct() for the meaning of size and num_freqs
        """
        
        self._filter_size = filter_size
        self._num_freqs = num_freqs
        
        filter_bank = dct(self._filter_size, self._num_freqs)
        super().__init__(filter_bank = filter_bank, feature_type = feature_type, 
                         mode = mode, gsconversion = gsconversion)   
        
    def __repr__(self):
        return super().__repr__() + '-{}px-{}'.format(self._filter_size, 
                                                      self._num_freqs)
        
class Gabor(ImageFiltering):
    """Image features based on Gabor filters"""
    
    def __init__(self, feature_type='MV', mode='SAME', stride=1, 
                 gsconversion='Luminance', size = 5, orientations=6, scales=5, 
                 scaling_factor=(2 ** 0.5), complex=True, sigma=None, 
                 lambda_=None, ellipt=1, phase=0):
        """The constructor
        
        Parameters
        ----------
        See ImageFiltering.__init__() for the meaning of feature_type, mode, 
            stride and gsconversion
        See gabor() for the meaning of size, orientations, scales, 
            scaling_factor, complex, sigma, lambda_, ellipt and phase 
        """
        
        self._size = size
        self._num_scales = scales
        self._num_orientations = orientations
        
        filter_bank = gabor(sz = self._size, 
                            orientations = self._num_orientations, 
                            scales = self._num_scales)
        super().__init__(filter_bank = filter_bank, feature_type = feature_type, 
                         mode = mode, gsconversion = gsconversion)   
        
    def __repr__(self):
        return super().__repr__() + '-{}px-{}-{}'.format(self._size,
                                                         self._num_scales, 
                                                         self._num_orientations)
 
class Laws(ImageFiltering):
    """Image features based on Laws masks"""
    
    def __init__(self, feature_type='MV', mode='SAME', stride=1, 
                 gsconversion='Luminance'):
        """The constructor
        
        Parameters
        ----------
        See ImageFiltering.__init__() for the meaning of feature_type, mode, 
            stride and gsconversione 
        """       
        
        filter_bank = laws()
        super().__init__(filter_bank = filter_bank, feature_type = feature_type, 
                         mode = mode, stride = stride, 
                         gsconversion = gsconversion) 
        
class Zernike(ImageFiltering):
    """Image features based on Zernike polynomials"""
    
    def __init__(self, feature_type='MV', mode='SAME', stride=1, 
                 gsconversion='Luminance', size=5, orders=6, include_odd=True):
        """The constructor
        
        Parameters
        ----------
        See ImageFiltering.__init__() for the meaning of feature_type, mode, 
            stride and gsconversione
        See zernike() for the meaning of size, orders and include_odd
        """ 
        
        self._size = size
        self._num_orders = orders
        self._include_odd = include_odd        
        
        filter_bank = zernike(sz = self._size, orders = self._num_orders, 
                              include_odd = self._include_odd)
        
        super().__init__(filter_bank = filter_bank, feature_type = feature_type, 
                         mode = mode, stride = stride, 
                         gsconversion = gsconversion) 
        
    def __repr__(self):
        return super().__repr__() + '-{}px-{}'.format(self._size,
                                                      self._num_orders)        


#****************************************
#****  Functions to generate filters ****
#****************************************
def separable_filters(singledim):
    """Combine a sequence of one-dimensional filters into a bank of 
    two-dimensional oness.

    Parameters
    ----------
    singledim : ndarray (sz, N) 
        One dimensional filters of size N

    Returns
    -------
    ndarray (N, N, M**2) 
        The two-dimensional filters
    """
    sz = singledim.shape[1]
    f = np.einsum('ai, bj -> ijab', singledim, singledim)
    return f.reshape([sz, sz, -1])

def gabor(sz, orientations=6, scales=5, scaling_factor=(2 ** 0.5),
          complex=True, sigma=None, lambda_=None,
          ellipt=1, phase=0, dtype=np.float32):
    """Gabor filters.

    Parameters
    ----------
    sz : int 
        Spatial size of the filters
    orientations : int
        Number of orientations
    scales : int
        Number of scales
    scaling_factor : float
        Ratio between consecutive scales
    complex : bool
        Whether to include filters corresponding to the imaginary part
    sigma : float 
        Shape of the Gaussuian envelope (default (sz - 1) / 5)
    lambda_ : float 
        Maximum wavelength (default sz)
    ellipt : float 
        Ellipticity of the filter. Use 1.0 for circular filters.
    phase : float
        Phase of the filter

    Returns
    -------
    - (sz, sz, orientations * scales * comps) array of filters
      where comps is 1 or 2 depending on the option complex.
    """

    if sigma is None:
        sigma = (sz - 1) / 5.0
    if lambda_ is None:
        lambda_ = sz
    pi = np.pi

    ax = np.arange(sz) - 0.5*(sz - 1)
    x, y = np.meshgrid(ax, ax)

    basis = np.empty((sz, sz, scales, orientations, 1 + complex),
                     dtype=dtype)
    for o in range(orientations):
        cos = math.cos(pi * o / orientations)
        sin = math.sin(pi * o / orientations)
        x1 =  x * cos + y * sin
        y1 = -x * sin + y * cos
        num = -(x1**2 + (ellipt * y1)**2)
        den = 2 * (sigma**2)
        g1 = np.exp(num / den)
        for s in range(scales):
            f = 2 * pi * (scaling_factor ** s) / lambda_
            g2 = np.cos(f * x1 + phase)
            basis[:,:,s,o,0] = g1 * g2
            if complex:
                g3 = np.sin(f * x1 + phase)
                basis[:,:,s,o,1] = g1 * g3
    return basis.reshape([sz, sz, -1])

def laws(dtype=np.float32):
    """Laws' masks 5x5
    
    ----------
    References
    [1] Laws K.I.
        Rapid texture identification
        Proc. SPIE 0238, Image Processing for Missile Guidance, 1980.
    
    Returns:
    - (5, 5, 25) array of 2D filters
    """
    coeffs = [
        [ 1,  4,  6,  4,  1],
        [-1, -2,  0,  4,  1],
        [-1,  0,  2,  0, -1],
        [-1,  2,  0, -2,  1],
        [ 1, -4,  6, -4,  1]
    ]
    return separable_filters(np.array(coeffs, dtype=dtype))


def dct(sz, n, dtype=np.float32):
    """Discrete cosine filters.

    Parameters
    ----------
    sz : int
        The size of the filter (in pixels)
    n  : int
        Number of horizontal and vertical frequencies

    Returns
    -------
    ndarray (sz, sz, n**2) 
        Array of discrete cosine filters
    
    References
    ----------
    [1]    Ahmed, N., Natarajan, T., Rao, K.R.
           Discrete Cosine Transform
           (1974) IEEE Transactions on Computers, C-23 (1), pp. 90-93.
    """
    x = 2 * np.arange(sz, dtype=dtype) + 1
    u = 0.5 * np.arange(n, dtype=dtype) * np.pi / sz
    p = np.cos(np.outer(u, x))
    return separable_filters(p)

def zernike(sz, orders, include_odd=True, dtype=np.float32):
    """Filters computing Zernike polynomials.

    There are Z = orders * (orders + 1) / 2 Zernike polynomials,
    divided into even and odd polynomials.

    Parameters
    ----------
    sz : int
        The size of the filters in pixels
    orders : int
        The number of orders of the filters
    include_odd : bool 
        Whether to include odd filters

    Returns
    -------
    ndarray (sz, sz, Z) 
        The Zernike polynomials
        
    References
    ----------
    [1]    Lakshminarayanan, V., Fleck, A.
           Zernike polynomials: A guide
           (2011) Journal of Modern Optics, 58 (7), pp. 545-561.
    """

    ax = np.linspace(-1, 1, sz, dtype=dtype)
    x, y = np.meshgrid(ax, ax)
    r = np.hypot(x, y)
    a = np.arctan2(x, y)
    unit_disk = disk(sz, 4, dtype=dtype)
    polys = _zernike_radial(orders, r)
    count = len(polys)
    if include_odd:
        count += sum(1 for (n, m, p) in polys if m > 0)
    basis = np.empty((sz, sz, count), dtype=dtype)
    index = 0
    for n, m, p in polys:
        basis[:,:,index] = unit_disk * p * np.cos(a * m)
        index += 1
        if include_odd and m > 0:
            basis[:,:,index] = unit_disk * p * np.sin(a * m)
            index += 1
    return basis

def _zernike_radial(orders, r):
    # Use a recurrence relation to compute the radial polynomials
    polys = {
        (0, 0): np.ones_like(r),
        (1, 1): r.copy()
    }
    for n in range(2, orders):
        polys[(n, n)] = r ** n
        polys[(n, n - 2)] = (n * (r ** 2) - (n-1)) * (r ** (n-2))
        for m in range(n % 2, n - 3, 2):
            p = 2 * (n - 1)
            p *= 2*n*(n - 2)*(r**2) - m**2 - n*(n-2)
            p *= polys[(n-2, m)]
            p -= n*(n + m - 2) * (n - m - 2) * polys[(n - 4, m)]
            polys[(n, m)] = p / (n + m) * (n - m) * (n - 2)
    return [(k[0], k[1], polys[k]) for k in sorted(polys)]

def disk(sz, up=1, dtype=np.float64):
    """Compute a unit disk.

    The elements in the disk have value 1, the others have value 0.
    The elments on the border can have intermediate values if
    smoothing is enabled by specifiying a value for up > 1.

    Parameters:
    - sz : size of the disk
    - up : upscaling used to smooth the disk

    Returns:
    - (sz, sz) array containing the disk

    """
    ax = np.linspace(-1, 1, up * sz, dtype=dtype)
    x, y = np.meshgrid(ax, ax)
    d = (np.hypot(x, y) <= 1)
    if up > 1:
        d = np.mean(d.reshape(sz, up, up * sz), axis=1)
        d = np.mean(d.reshape(sz, sz, up), axis=2)
    return d