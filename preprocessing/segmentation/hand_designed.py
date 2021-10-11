from math import log2

from abc import abstractmethod
import cv2
import matplotlib.pyplot as plt
import numpy as np

from scipy import ndimage as ndi

from skimage import data, img_as_float
from skimage.filters import rank, threshold_otsu, threshold_multiotsu
from skimage.measure import label
from skimage.morphology import disk
from skimage.segmentation import felzenszwalb, flood_fill,\
     morphological_chan_vese, morphological_geodesic_active_contour,\
     slic,watershed 

from sklearn.cluster import KMeans as KMeans_

from cenotaph.basics.base_classes import Image, ImageHandler, ImageType
from cenotaph.third_parties.doc_inherit import doc_inherit

from utilities import binary_mask

class ImageSegmenter(ImageHandler):
    """Abstract base class for image segmentation"""
    
    @abstractmethod 
    def _segment(self, img, **kwargs):
        """Abstract method to be implemented by the subclasses.
        
        Parameters
        ----------
        img : Image
            The input image. 

        Returns
        ----------
        segmented : ndarray
            The segmented image. This is a greyscale image where each grey
            level corresponds to a different resgion. The size is the same as
            the input image.
        """
    
    def _check_type(self, img):
        """The input image needs to be RGB or greyscale. 
               
        Parameters
        ----------
        img : Image
            The image to segment
            
        Returns
        -------
        check_ok : bool
            True if img is the right type, false otherwise.
        """
        check_ok = True
        if img.get_type() not in [ImageType.RGB, ImageType.GS]:
            check_ok = False
        
        return check_ok
    
    def get_result(self, img, **kwargs):
        """
        Get the segmentation result
        
        Parameters
        ----------
        img : Image
            The grey-scale input image to segment. 
        seed_pixel : int (2), optional
            Converts the result of the segmentation to a binary image where
            all the pixels that have the same value as seed_pixel are set
            to (2**bit_depth - 1), the others are set to zero.
        excluded_values : list of positive int (optional)
            Do not do the filtering if the value of the seed pixels is in
            excluded_values. 

        Returns
        ----------
        img_out : Image
            The segmented image. This is a greyscale image where each grey
            level corresponds to a different region.
        """
        
        #Check that the type of the input image is the correct one for the
        #segmentation method chosen
        if not self._check_type(img):
            raise Exception('The input image is not compatible with the'
                            'segmenter')
                
        #Call the class-specific segmenter and return the result
        data_out = self._segment(img, **kwargs)
        
        #Set the right data type depending on the number of levels returned
        #by the segmenter
        num_bits, data_type = self.get_encoding(np.amax(data_out[:]) + 1)
        data_out = data_out.astype(data_type)
        
        #Generate the output image
        img_out = Image.from_array(data = data_out, img_type = ImageType.GS)
        
        #Filter on the pivot pixel if required
        if 'seed_pixel' in kwargs.keys():
            if 'excluded_values' in kwargs.keys():
                img_out = self.filter_by_pixel(
                    img = img_out, 
                    seed_pixel = kwargs['seed_pixel'],
                    excluded_values = kwargs['excluded_values'])
            else:
                img_out = self.filter_by_pixel(
                    img = img_out, 
                    seed_pixel = kwargs['seed_pixel'])                
                
        return img_out
    
    def get_short_name(self):
        """Short representation
        
        Returns
        -------
        short_name : str
        """
        return self.__class__.__name__
        
    
    @staticmethod
    def init_ls_digital_circle(img, center, radius, exp):
        """Initial level set as a digital circle.
        
        Parameters
        ----------
        img : Image
            The image to segment.
        center : (int,int)
            The coordinates of the center of the digital circle.
        radius : int
            The radius of the digital circle.
        exp : int
            The exponent of the Minkowsky distance. Use 1 for Manhattan 
            distance, 2 for Euclidean distance, etc.
        
        Returns
        -------
        init_ls : Image
            A binary mask the same size of the input image with a bright
            digital circle. 
        """
        
        [H,W] = img.get_size()
        init_ls = binary_mask(size = (H,W), 
                              center = center, 
                              radius = radius,
                              exp = exp)  
        return init_ls
    
    @staticmethod
    def filter_by_pixel(img, seed_pixel, excluded_values=[]):
        """Filter the result of the segmentation based on a given pixel.
        The function determines the connected regions in the input image and
        filters out all the regions not containing the seed pixel. 
        All the pixels within the region containing the seed pixel are set to 1, 
        the others to the maximum.
        
        Parameters
        ----------
        img : Image
            A greyscale image, usually is the result of the segmentation.
        pixel : int (2)
            The coordinates of the pixel.
        excluded_values : list of positive int (optional)
            Do not do the filtering if the value of the seed pixels is in
            excluded_values.
            
        Returns
        -------
        img_out : Image
            A binary image.
        """
        
        #Find the connected regions
        regions = label(input = img.get_data(), 
                        background = 0, 
                        connectivity = 1)
        labels = np.unique(regions)  
        
        #Get the value of the seed pixel as the pivot_value
        pivot_value = regions[seed_pixel[0],seed_pixel[1]]
        
        
        if pivot_value not in excluded_values:
        
            #Do the filtering
            new_data = img.get_data(copy = True)
            new_data[:] = 0
            new_data[np.where(regions == pivot_value)] = 1        
        
            #Create and return the output image
            img_out = Image.from_array(data = new_data, img_type = ImageType.BW)
        else:
            img_out = img
            
        return img_out
        
    @staticmethod
    def get_encoding(num_levels):
        """Get the correct encoding (inner type) given the number of levels
        
        Parameters
        ----------
        num_levels : int
            The number of levels
            
        Returns
        -------
        num_bits : int
            The number of bits required for encoding the given number of 
            levels. Can be 8, 16, 32 or 64
        nptype : numpy numeric type
            The numpy type required for encoding the given number of levels. 
            Can be: np.uint8, np.uint16, np.uint32 or np.uint64
        """
        
        #Look-up table for bit depth and related type
        bit_depth_lut = {8 : np.uint8, 16 : np.uint16, 32 : np.uint32,
                         64 : np.uint64}
        
        #Compute the number of bits required for encoding num_levels
        bits = np.log2(num_levels)
        
        #Get the closest value in the lut
        values = list(bit_depth_lut.keys())
        distances = abs(values - bits)
        indices = np.argsort(distances)
        
        num_bits = values[indices[0]]
        nptype = bit_depth_lut[num_bits]
        
        return num_bits, nptype
        
        

class ImageSegmenterGS(ImageSegmenter):
    """Abstract base class for image segmentation of grey-scale images"""
    
    @doc_inherit
    def _check_type(self, img):
        """Return True if the image is greyscale, False otherwise"""
        
        check_ok = True
        if not (img.get_type() == ImageType.GS):
            check_ok = False  
        
        return check_ok
    
    @staticmethod
    def greyscale_histogram(img):
        """Greyscale histogram
        
        Parameters
        ----------
        img : Image
            The input image (needs to be greyscale)
            
        Returns
        -------
        hist : ndarray of float (N)
            The normalised histogram (sum = 1.0).
        edges : ndarray od float (N+1)
            The histogram edges.
        """
        
        if not (img.get_type() == ImageType.GS):
            raise Exception('The input image needs to be greyscale')
        
        n = img.get_bit_depth()
        data_in = img.get_data()
        
        #Compute the grey-level histogram
        histogram_bounds = (-0.5, 2**n + 0.5)
        edges = np.arange(start = histogram_bounds[0],
                          stop = histogram_bounds[1])
        edges = np.append(edges, values = [histogram_bounds[1]])
        hist, _ = np.histogram(data_in[:], bins = edges) 
        
        return hist, edges
    
class MorphACWE(ImageSegmenterGS):
    """Image segmentation based on morphological active contours without edges
    
    References
    ----------
    [1] Chan, T.F., Vese, L.A. Active contours without edges (2001) 
        IEEE Transactions on Image Processing, 10 (2), pp. 266-277.
    [2] Marquez-Neila, P., Baumela, L., Alvarez, L. A morphological approach 
        to curvature-based evolution of curves and surfaces (2014) 
        IEEE Transactions on Pattern Analysis and Machine Intelligence, 36 (1), 
        art. no. 6529072, pp. 2-17.
    """
    
    def __init__(self, n_iter=10, lambda_1=1.0, lambda_2=1.0, init_ls_radius=3, 
                 init_ls_distance_exp=2):
        """
        Parameters
        ----------
        n_iter : int
            The number of iterations to run.
        lambda_1 : float (> 0.0)
            Weight parameter of the inner region in the objective function.
        lambda_2 : float (> 0.0)
            Weight parameter of the outer region in the objective function.
        init_ls_distance_exp : int
            The exponent of the Minkowsky distance used to create the initial
            level set. Use 1 for Manhattan distance, 2 for Euclidean distance,
            etc.
        """     
        
        self._n_iter = int(round(n_iter))
        self._lambdas = (lambda_1, lambda_2)
        self._init_ls_radius = init_ls_radius
        self._init_ls_distance_exp = init_ls_distance_exp        
    
    @doc_inherit   
    def _segment(self, img, **kwargs):
        
        #Get the seed pixel
        try:
            seed_pixel = kwargs['seed_pixel']
        except KeyError:
            raise Exception('Requires a seed pixel')
         
        #Create the initial level set
        init_ls = self.init_ls_digital_circle(
            img, center = seed_pixel, radius = self._init_ls_radius,
            exp = self._init_ls_distance_exp)                                       
        
        segmented = morphological_chan_vese(image = img.get_data(), 
                                            iterations = self._n_iter, 
                                            init_level_set = init_ls.get_bwimage(), 
                                            smoothing = 1, 
                                            lambda1 = self._lambdas[0], 
                                            lambda2 = self._lambdas[1])
        
        return segmented    

                
class MorphGAC(ImageSegmenterGS):
    """Image segmentation based on Morphological Geodesic Active Contours
    
    References
    ----------
    [1] Caselles, V., Kimmel, R., Sapiro, G. Geodesic Active Contours (1997) 
        International Journal of Computer Vision, 22 (1), pp. 61-79.
    [2] Marquez-Neila, P., Baumela, L., Alvarez, L. A morphological approach 
        to curvature-based evolution of curves and surfaces (2014) 
        IEEE Transactions on Pattern Analysis and Machine Intelligence, 36 (1), 
        art. no. 6529072, pp. 2-17.
    """
    
    def __init__(self, n_iter=10, baloon_force=0.0, init_ls_radius=3, 
                 init_ls_distance_exp=2):
        """
        Parameters
        ----------
        n_iter : int
            The number of iterations to run.
        baloon_force : float
            Balloon force to guide the contour in non-informative areas of the
            image where the gradient is too small to push the contour towards 
            a border. Negative values will shrink the contour, positive ones 
            will expand it. Setting this to zero disables the balloon force.
        init_ls_radius : int
            The radius of the initial level set that will be created around
            the seed point.
        init_ls_distance_exp : int
            The exponent of the Minkowsky distance used to create the initial
            level set. Use 1 for Manhattan distance, 2 for Euclidean distance,
            etc.
        """
        self._n_iter = int(round(n_iter))
        self._baloon_force = baloon_force
        self._init_ls_radius = init_ls_radius
        self._init_ls_distance_exp = init_ls_distance_exp
                    
    @doc_inherit   
    def _segment(self, img, **kwargs):
        
        #Get the seed pixel
        try:
            seed_pixel = kwargs['seed_pixel']
        except KeyError:
            raise Exception('Requires a seed pixel')
         
        #Create the initial level set
        init_ls = self.init_ls_digital_circle(
            img, center = seed_pixel, radius = self._init_ls_radius,
            exp = self._init_ls_distance_exp)                                          
                                            
        segmented = morphological_geodesic_active_contour(
            img_as_float(img.get_data()), iterations=self._n_iter, 
            init_level_set = init_ls.get_bwimage(), 
            smoothing = 1, threshold='auto', 
            balloon = self._baloon_force)
        
        return segmented
    
    def __repr__(self):
        return self.__class__.__name__ +\
               ' num. iter. = {}, baloon force = {:4.2f}'.\
               format(self._n_iter, self._baloon_force)
        
class Watershed(ImageSegmenterGS):
    """Image segmentation based on watershed transform with compactness
    constraint.
    
    References
    ----------
    [1] https://en.wikipedia.org/wiki/Watershed_%28image_processing%29
    [2] http://cmm.ensmp.fr/~beucher/wtshed.html
    [3] Neubert, P., Protzel, P. Compact watershed and preemptive SLIC: On 
        improving trade-offs of superpixel segmentation algorithms (2014) 
        Proceedings - International Conference on Pattern Recognition,
        art. no. 6976891, pp. 996-1001. 
    """  
    
    def __init__(self, markers='auto', compactness=0.0, radius=2):
        """
        Parameters
        ----------
        markers : Image or str
            'auto' -> Markers are computed automatically on the gradient image
                      by thresholding this at the median level.
            Image  -> An array marking the basins with the values to be assigned 
                      in the label matrix. Zero means not a marker. 
                      Needs to have the same size as the image to segment.
        compactness : float
            Use compact watershed [3] with given compactness parameter. Higher
            values result in more regularly-shaped watershed basins.
        radius : int
            Radius of the structuring element (disc) used to compute the 
            gradient.
        """   
        self.set_markers(markers)
        self._compactness = compactness
        self._radius = radius
        a = 0
        
    def _segment(self, img, **kwargs):
        
        #Compute the gradient image
        gradient_img = rank.gradient(img.get_data(), disk(self._radius))
        
        #Extract the markers
        if self._markers == 'auto':
            markers = gradient_img < np.median(gradient_img)
            markers = ndi.label(markers)[0] 
        else:
            if not (self._markers.get_size() == img.get_size()):
                raise Exception('The markers and the input image must have the'
                                'same size')            
            markers = self._markers.get_data()        
                        
        segmented = watershed(gradient_img, 
                              markers = markers, 
                              connectivity = 1, offset = None, mask = None, 
                              compactness = self._compactness, 
                              watershed_line = False)
        
        return segmented
        
    def set_markers(self, markers):
        """Set the markers
        
        Parameters
        ----------
        markers : Image
            An array marking the basins with the values to be assigned in the 
            label matrix. Zero means not a marker. Need to have the same
            size as the image to segment.
        """
        self._markers = markers
        
    def __repr__(self):
        return self.__class__.__name__ +\
               ' compactness = {:4.2f}'.format(self._compactness) +\
               ' radius = {:4.2f}'.format(self._radius)
 
class Felzenszwalb(ImageSegmenterGS):
    """ Felsenszwalb’s graph-based image segmentation. 
    
    References
    ----------
    [1] Felzenszwalb, P.F., Huttenlocher, D.P. Efficient graph-based image 
    segmentation (2004) International Journal of Computer Vision, 59 (2), 
    pp. 167-181. 
    """
    
    
    def __init__(self, scale = 1.0, sigma = 0.8, min_size = 5):
        """
        Parameters
        ----------
        scale : float
            Free parameter: higher values mean larger clusters.
        sigma : float
            Width (standard deviation) of the Gaussian kernel used for
            image preprocessing.
        min_size : int
            Minimum size of the smallest region. Enforced using postprocessing.
        """ 
        self._scale = scale
        self._sigma = sigma
        self._min_size = int(round(min_size))
    
    def _segment(self, img, **kwargs):
        segmented = felzenszwalb(image = img.get_data(), 
                                 scale = self._scale, 
                                 sigma = self._sigma,
                                 min_size = self._min_size)
        return segmented
    
    def __repr__(self):
        return self.__class__.__name__ +\
               ' scale = {:4.2f}, sigma = {:4.2f}, min size = {}'.\
               format(self._scale, self._sigma, self._min_size)    
            
class MSER(ImageSegmenterGS):
    """Maximally stable extremal regions
    
    References
    ----------
    [1] Matas, J., Chum, O., Urban, M., Pajdla, T. Robust wide-baseline stereo
        from maximally stable extremal regions (2004) Image and Vision 
        Computing, 22 (10 SPEC. ISS.), pp. 761-767.
    """
    
    def __init__(self, delta=5, min_area=10, max_area=0.25, 
                 max_variation = 0.25):
        """
        Parameters
        ----------
        delta : int [0, 2**nbits - 1]
            Indicates how many different gray levels does a region need to be 
            stable to be considered maximally stable. Higher values will result
            in fewer regions
        min_area : int
            The lower cut-off value for the region size. Regions with an area 
            (in pixels) below the threshold will be excluded.
        max_area : float
            The upper cut-off value for the region size as a fraction of the
            total area of the input image. 
        max_variation : float
            If a region is maximally stable, it can still be rejected if its 
            variation (in area) is bigger than max_variation.
        """
        
        self._delta = int(round(delta))
        self._min_area = min_area
        self._max_area = max_area
        self._max_variation = max_variation
            
    def _segment(self, img, **kwargs):
        
        img_size = img.get_size()
        img_area = img_size[0] * img_size[1]
        
        #Create the MSER detector
        mser_detector = cv2.MSER_create(_delta = self._delta, 
                                        _min_area = self._min_area, 
                                        _max_area = round(self._max_area * img_area), 
                                        _max_variation = self._max_variation, 
                                        _min_diversity = 0.2, 
                                        _max_evolution = 200, 
                                        _area_threshold = 1.01, 
                                        _min_margin = 0.003, 
                                        _edge_blur_size = 0)        
        
        #Detect the MSER regions
        regions_coordinates, _ = mser_detector.detectRegions(img.get_data())
        
        #Compute the number of bits required to encode the regions
        num_regions = len(regions_coordinates)
        num_bits = log2(num_regions)
        
        #Round to 8, 16, 32 or 64 bits
        round_to = np.array([8, 16, 32, 64])
        types = [np.uint8, np.uint16, np.uint32, np.uint64]
        abs_diff = abs(round_to - num_bits) 
        sorted_indices = np.argsort(abs_diff)
        
        #Create empty output with required data type
        data_out = np.zeros(img.get_data().shape, 
                            dtype = types[sorted_indices[0]])
        
        #Apply a different label to each region (zero for unlabelled pixels)
        for r, region in enumerate(regions_coordinates):
            for px in region:
                data_out[px[0],px[1]] = r + 1
        
        return data_out   
        
    def __repr__(self):
        return self.__class__.__name__ +\
               ' delta = {}, min area = {}, max area = {}, max variation = {}'.\
               format(self._delta, self._min_area, 
                      self._max_area, self._max_variation) 
    
class FloodFill(ImageSegmenterGS):
    """Segmentation based on the flood-fill algoritmh"""
    
    def __init__(self, delta=2, connectivity=2):
        """
        Parameters
        ----------
        delta : int
            Adjacent points are filled if their values are within plus or minus 
            delta/2 from the seed point are filled.
        connectivity : int
            Adjacent pixels with squared distance from the center less than or 
            equal to connectivity are considered neighbours.
        """
        self._delta = delta
        self._connectivity = connectivity
        
    def _segment(self, img, **kwargs):
        
        #Store the original data type
        data_in = img.get_data()
        original_dtype = data_in.dtype                
                
        #Get the input data as signed int
        data_in = data_in.astype(np.int)
        
        #Apply flood fill
        marker_value = -1
        data_out = flood_fill(image = data_in, 
                              seed_point = kwargs['seed_pixel'], 
                              new_value = marker_value, 
                              connectivity = self._connectivity, 
                              tolerance = self._delta/2)
        
        #Set to 1 all the pixels in the fill and 0 the others
        data_out[np.where(data_out != marker_value)] = 0
        data_out[np.where(data_out == marker_value)] = 1
        
        #Convert back to the original data type and return
        data_out = data_out.astype(original_dtype)
        
        return data_out
                
        
class GlobalThresholding(ImageSegmenterGS):
    """Base class for segmentation methods based on global thresholding"""
    
    @abstractmethod
    def compute_thresholds(self, img, **kwargs):
        """Compute the threshold(s). The core function that needs to be 
        implemented by each subclass. 
        
        Parameters
        ----------
        img : Image
            The input image (needs to be greyscale)
        num_thresholds : int, optional
        
        Returns
        -------
        thresholds : list of numeric
            The thresholds.
        """
    
    @doc_inherit
    def _segment(self, img, **kwargs):   
        data_in = img.get_data()
        thresholds = self.compute_thresholds(img, **kwargs)
        segmented = self.do_thresholding(data_in, thresholds)
        return segmented     
    
    @staticmethod
    def do_thresholding(data_in, t):
        """Threshold the input data
        
        Parameters
        ----------
        input_data : nparray of numeric (H,W)
            The data to threshold.
        t : list of numeric (T)
            The threshold values in ascending order [t0,t1,...,t_{n-1}].
        
        Returns
        -------
        data_out : nparray of uint8 (H,W)
            The thresholded data with values in the [0,n-1] interval:  
                0 for X in [0,t0[ 
                1 for X in [t0,t1[
                ...
                n-1 for X in [t_n,t_{n-1}]
        """
        data_out = np.zeros(data_in.shape).astype(np.uint8)
        for assigned_value, threshold_value in enumerate(t):            
            data_out[np.where(data_in >= threshold_value)] = assigned_value + 1
        return data_out

class Otsu(GlobalThresholding):
    """Segmentation based on Otsu's thresholding method
    
    References
    ----------
    [1] Otsu, Nobuyuki
        Threshold selection method from gray-level histograms.
        (1979) IEEE Trans Syst Man Cybern, SMC-9 (1), pp. 62-66. 
    """
    
    @doc_inherit
    def compute_thresholds(self, img, **kwargs):
        return [threshold_otsu(img.get_data(), 
                               nbins = 2**img.get_bit_depth())]
    
class MultiOtsu(GlobalThresholding):
    """Segmentation based on Otsu's multi-level thresholding method
    
    References
    ----------
    [1] Liao, P.-S., Chen, T.-S., Chung, P.-C.
        A fast algorithm for multilevel thresholding (2001)
        Journal of Information Science and Engineering, 17 (5), pp. 713-727. 
    """
    
    def __init__(self, num_classes):
        """
        Parameters
        ----------
        num_classes : int (> 2)
            The number of classes into which partition the original image. The
            number of thresholds will be num_classes - 1.
        """
        self._num_classes = int(num_classes)
    
    @doc_inherit
    def compute_thresholds(self, img, **kwargs):
        return threshold_multiotsu(img.get_data(), classes = self._num_classes)
        
class Kittler(GlobalThresholding):
    """Segmentation based on Kittler-Illingworth method
    
    References
    ----------
    [1] Kittler, J., Illingworth, J.
        On Threshold Selection Using Clustering Criteria
        (1985) IEEE Transactions on Systems, Man and Cybernetics, SMC-15 (5), 
        pp. 652-655.
    """ 
    
    @doc_inherit
    def compute_thresholds(self, img, **kwargs):
        return [self.min_err_threshold(img)]    
        
    @staticmethod
    def min_err_threshold(img):
        """Minimum error thresholding algorithm.
    
        Parameters
        ----------
        img : Image
            The input image.
        bit_depth : uint

        Returns
        -------
        threshold : int
            The computed threshold.
        
        Credits
        -------
        Modified version of the function available in pythreshold to
        accept input image with arbitrary bit depth
        
        """
        # Input image histogram
        hist, _ = ImageSegmenterGS.greyscale_histogram(img)
        #hist = np.histogram(image, bins=range(256))[0].astype(np.float)
    
        # The number of background pixels for each threshold
        w_backg = hist.cumsum()
        w_backg[w_backg == 0] = 1  # to avoid divisions by zero
    
        # The number of foreground pixels for each threshold
        w_foreg = w_backg[-1] - w_backg
        w_foreg[w_foreg == 0] = 1  # to avoid divisions by zero
    
        # Cumulative distribution function
        cdf = np.cumsum(hist * np.arange(len(hist)))
    
        # Means (Last term is to avoid divisions by zero)
        b_mean = cdf / w_backg
        f_mean = (cdf[-1] - cdf) / w_foreg
    
        # Standard deviations
        b_std = ((np.arange(len(hist)) - b_mean)**2 * hist).cumsum() / w_backg
        f_std = ((np.arange(len(hist)) - f_mean) ** 2 * hist).cumsum()
        f_std = (f_std[-1] - f_std) / w_foreg
    
        # To avoid log of 0 invalid calculations
        b_std[b_std == 0] = 1
        f_std[f_std == 0] = 1
    
        # Estimating error
        error_a = w_backg * np.log(b_std) + w_foreg * np.log(f_std)
        error_b = w_backg * np.log(w_backg) + w_foreg * np.log(w_foreg)
        error = 1 + 2 * error_a - 2 * error_b
    
        return np.argmin(error)
    
    
class Kapur(GlobalThresholding):
    """Segmentation based on max entropy thresholding method
    
    References
    ----------
    [1] Kapur, J.N., Sahoo, P.K., Wong, A.K.C.
        A new method for gray-level picture thresholding using the entropy 
        of the histogram. (1985) Computer Vision, Graphics, & Image 
        Processing, 29 (3), pp. 273-285. 
    """
    
    @doc_inherit
    def compute_thresholds(self, img, **kwargs):
        
        #Compute the histogram
        hist, _ = self.greyscale_histogram(img)
                
        #Compute the threshold
        t = self.max_entropy(hist)        
        
        return [t]    
        
    @staticmethod
    def max_entropy(data):
        """
        Parameters
        ----------
        data : Sequence representing the histogram of the image
        
        Returns
        -------
        threshold : Resulting maximum entropy threshold
                
        Credits
        -------
        Source: https://github.com/zenr/ippy/blob/master/segmentation/max_entropy.py
        
        Ported to ImageJ plugin by G.Landini from E Celebi's fourier_0.8 routines
        2016-04-28: Adapted for Python 2.7 by Robert Metchev from Java source 
        of MaxEntropy() in the Autothresholder plugin
        http://rsb.info.nih.gov/ij/plugins/download/AutoThresholder.java
        """
    
        # calculate CDF (cumulative density function)
        cdf = data.astype(np.float).cumsum()
    
        # find histogram's nonzero area
        valid_idx = np.nonzero(data)[0]
        first_bin = valid_idx[0]
        last_bin = valid_idx[-1]
    
        # initialize search for maximum
        max_ent, threshold = 0, 0
    
        for it in range(first_bin, last_bin + 1):
            # Background (dark)
            hist_range = data[:it + 1]
            hist_range = hist_range[hist_range != 0] / cdf[it]  # normalize within selected range & remove all 0 elements
            tot_ent = -np.sum(hist_range * np.log(hist_range))  # background entropy
    
            # Foreground/Object (bright)
            hist_range = data[it + 1:]
            # normalize within selected range & remove all 0 elements
            hist_range = hist_range[hist_range != 0] / (cdf[last_bin] - cdf[it])
            tot_ent -= np.sum(hist_range * np.log(hist_range))  # accumulate object entropy
    
            # find max
            if tot_ent > max_ent:
                max_ent, threshold = tot_ent, it
    
        return threshold
    
class KMeans(ImageSegmenter):
    """K-means clustering
    
    References
    ----------
    [1] Arthur, D., Vassilvitskii, S. 
        K-means++: The advantages of careful seeding (2007) 
        Proceedings of the Annual ACM-SIAM Symposium on Discrete Algorithms, 
        07-09-January-2007, pp. 1027-1035.
    """
    
    def __init__(self, num_classes=3):
        """
        Parameters
        ----------
        num_classes : int
            The number of classes
        """
        self._num_classes = int(num_classes)
        self._clusterer = KMeans_(n_clusters = self._num_classes, 
                                  random_state = 0)
    
    @doc_inherit
    def _segment(self, img, **kwargs):  
        
        #Flatten the data and do the clustering
        data_in = img.get_data() 
        flat_data = data_in.flatten(order = 'F')
        kmeans = self._clusterer.fit(flat_data.reshape(-1,1))
        
        #Note: reshape(-1,1) needed because sklearn.cluster.KMeans can't cope
        #with 1d data!?
        
        #Get the labels
        segmented = kmeans.labels_
        
        #Reshape the data into the original format
        segmented = np.reshape(segmented, newshape = data_in.shape, order='F')
        
        return segmented    
    
class SLIC(ImageSegmenter):
    """ Simple Linear Iterative Clustering (SLIC). 
    
    References
    ----------
    [1] Achanta, R., Shaji, A., Smith, K., Lucchi, A., Fua, P., Süsstrunk, S.
        SLIC superpixels compared to state-of-the-art superpixel methods
        (2012) IEEE Transactions on Pattern Analysis and Machine Intelligence,
        34 (11), art. no. 6205760, pp. 2274-2281. 
    """ 
        
    def __init__(self, n_segments=10, compactness=1.0):
        """
        Parameters
        ----------
        n_segments : int
            The (approximate) number of labels in the segmented output image.
        compactness : float
            Balance between color/greyscale proximity and space proximity. 
            Higher values give more weight to space proximity, making 
            superpixel shapes more square/cubic. 
        """
        self._n_segments = n_segments
        self._compactness = compactness
    
    @doc_inherit
    def _segment(self, img, **kwargs):           
        segmented = slic(img.get_data(), n_segments = self._n_segments, 
                         compactness = self._compactness)
        return segmented

    def __repr__(self):
        return self.__class__.__name__ +\
               ': num segments = {}, compactness = {:4.2f}'.\
               format(self._n_segments, self._compactness)     
        