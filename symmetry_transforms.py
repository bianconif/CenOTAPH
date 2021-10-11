from cenotaph.basics.neighbourhood import Neighbourhood
from cenotaph.basics.neighbourhood import SquareNeighbourhood
from cenotaph.basics.generic_functions import combine_patterns
from cenotaph.basics.matrix_displaced_copies import matrix_displaced_copies
from cenotaph.basics.base_classes import *

class SymmetryTransform(ImageDescriptorGS):
    """LOCuST: LOCal Symmetry Transform. Abstract base class"""
    
    def __init__(self, imgfile, resolution=3, geometric_transform='r90', \
                 statistical_test='distance (L1)', *args):
        """
        Default constructor
        
        Parameters
        ----------
        imgfile    : str
            The path to the input image.
        resolution : odd int
            The dimension (side length) of the square neighbourhood.
        geometric_transform : str
            The geometric transform to check symmetry for. Can be:
                'r90'       -> counter-clockwise rotation by 90°
                'r180'      -> counter-clockwise rotation by 180°
                'r270'      -> counter-clockwise rotation by 270°
                'hMirror'   -> mirror about the horizontal mid-line
                'vMirror'   -> mirror about the vertical mid-line
                'd1Mirror'  -> mirror about the bottom right/top left diagonal
                'd2Mirror'  -> mirror about the bottom left/top right diagonal
        statistical_test : str
            The statistical test used for checking symmetry. Can be:
                'L2-distance-std' -> Euclidean distance with prelimiary zero mean 
                                     and unit variance standardisation.
        args : int
            Entity of the displacement (in pixels). Required if geometri_transform
            is a translation
        """
    
        if not ((resolution % 2) == 1):
            Exception('Resolution needs to be an odd number')
            
        super().__init__(imgfile)
        
        self._resolution = resolution
        self._geometric_transform = geometric_transform
        self._statistical_test = statistical_test
        self._updated = False
        self._map = np.empty(self._img.size)
            
        #Generate the original and the transformed neighbourhood (wich are the same 
        #at the beginning)
        self._original_neighbourhood = SquareNeighbourhood(resolution)
        self._transformed_neighbourhood = SquareNeighbourhood(resolution)
            
    def _compute_features(self):
        """Dummy implementation"""
        self._features = np.empty([0])
        
    def _compute_symmetry_map(self):
        """Compute the symmetry map"""
    
        #Generate the tranformed patterns
        self._apply_transform()
        self._transformed_patterns = \
        matrix_displaced_copies(self._img, \
                                self._transformed_neighbourhood.get_integer_points()) 
        
        #Carry out the comparison between the original and the transformed patterns
        self._perform_test()
        
        self._updated = True        
    
    def get_symmetry_map(self):
        """The resulting symmetry map
        
        Returns
        -------
        _map : ndarray
            A matrix the same size of the input image representing the symmetry 
            map
        """
        
        if not self._updated:
            self._compute_symmetry_map()
            self._updated = True
            
        return self._map
    
    def _apply_transform(self):
        """Apply the geometric transformation to the neighbourhood of points"""
        
        if self._geometric_transform == 'r90':
            self._transformed_neighbourhood.rotate(90) 
        elif self._geometric_transform == 'r180':
            self._transformed_neighbourhood.rotate(180)
        elif self._geometric_transform == 'r270':
            self._transformed_neighbourhood.rotate(270)   
        elif self._geometric_transform == 'hMirror':
            self._transformed_neighbourhood.reflect(1.0, 0.0, 0.0)
        elif self._geometric_transform == 'vMirror':
            self._transformed_neighbourhood.reflect(0.0, 1.0, 0.0) 
        elif self._geometric_transform == 'd1Mirror':
            self._transformed_neighbourhood.reflect(1.0, 1.0, 0.0)
        elif self._geometric_transform == 'd2Mirror':
            self._transformed_neighbourhood.reflect(-1.0, 1.0, 0.0) 
        else:
            raise Exception('Geometric transform not supported') 
                                       
    def _perform_test(self, neglect_fixed_points=True):
        """Perform the statistical test between the original and the transformed
        patterns
        
        Parameters
        ----------
            neglect_fixed_points : bool
                If True the fixed points are not considered in the test
        """
        
        original_patterns = np.empty([])
        transformed_patterns = np.empty([])
        
        if neglect_fixed_points:
            #Compute the fixed points
            fixed_points = Neighbourhood.compare(self._original_neighbourhood,
                                                 self._transformed_neighbourhood) 
            
            #Compute the original and transformed neighbourhoods without fixed 
            #points
            original_neighbourhood_no_fixed_points = \
                Neighbourhood.from_points(self._original_neighbourhood.get_points())
            original_neighbourhood_no_fixed_points.delete_points(fixed_points)
            
            transformed_neighbourhood_no_fixed_points = \
                Neighbourhood.from_points(self._transformed_neighbourhood.get_points())
            transformed_neighbourhood_no_fixed_points.delete_points(fixed_points)           
            
            #Generate the original patterns
            original_patterns = \
            matrix_displaced_copies(self._img, \
                                    original_neighbourhood_no_fixed_points.get_integer_points())             
            
            #Generate the transformed patterns
            transformed_patterns = \
            matrix_displaced_copies(self._img, \
                                    transformed_neighbourhood_no_fixed_points.get_integer_points())
        else:
            #Generate the original patterns
            original_patterns = \
            matrix_displaced_copies(self._img, \
                                    self._original_neighbourhood.get_integer_points())             
            #Generate the transformed patterns
            transformed_patterns = \
            matrix_displaced_copies(self._img, \
                                    self._transformed_neighbourhood.get_integer_points())            
        
        if self._statistical_test == 'L2-distance-std':
            self.standardise()
            self._map = combine_patterns(original_patterns, \
                                         transformed_patterns, \
                                         'distance', \
                                         n=1)
        else:
            raise Exception('Statistical test not supported')
        