import numpy as np
import cenotaph.basics.generic_functions as gf
from cenotaph.preprocessing.colour.base_classes import ColourPreprocessor
from copy import deepcopy

class PCAPerturbation():
    """Base class for defining a perturbation based on the principal components
    of the colour distribution"""
    
    def __init__(self, img):
        """
        Parameters
        ----------
        img : Image
            The input image
        """
        
        self._img_in = img
        
        #Get and store the number of pixels of the input image                            
        self._num_pixels = self._img_in.get_num_pixels()
        
        #Reshape the input image from rows x cols x 3 to (rows x cols) x 3 and
        #store the result
        self._data = self._img_in.get_data()
        self._reshaped_data = gf.image_reshape(self._data)
                
        #Assign 1/N weight to each pixel -> total mass of the colour distr. = 1.0
        _mass_distr = np.ones(self._num_pixels)/self._num_pixels    
        
        #Compute the eigenvalues and eigenvectors of the colour distribution
        _, self._eigvals, self._eigvecs = gf.inertia_matrix(self._reshaped_data, \
                                                            _mass_distr)
        
        #Compute the radii of gyration
        self._radii_of_gyration = np.sqrt(self._eigvals)
        
        #Compute the principal components
        self._princomps = np.multiply(self._eigvecs, \
                                      np.tile(self._radii_of_gyration, (self._eigvals.shape[0], 1)))         
        
        
class PCATranslationalPerturbation(PCAPerturbation):
    """Translational perturbation for colour augmentation based on the principal
    components of the colour distribution"""
    
    def __init__(self, img, disp_values):
        """
        Parameters
        ----------
        img : Image
            The input image
        disp_values : ndarray of float (3)
            The value of the displacements as a fraction of the radii of gyration
            of the corresponding principal axes. For example, let rI, rII and rIII 
            indicate the radii of gyration along the 1st, 2nd a 3rd principal axes.
            Values = (0.1, 0.3, -0,5) will indicate a translation of 
            0.1*r1 along I, 0.3*r2 along II and -0.5*r3 along III 
        """
                
        #Make sure the number of supplied displacements is correct
        if not (len(disp_values) == 3):
            raise Exception('The number of supplied displacements is incorrect')
        
        #Invoke the superclass constructor
        super().__init__(img)
        
        self._disp_values = disp_values
            
    def apply(self):
        """Apply the perturbation
        
        Parameters
        ----------
        data_in : ndarray of int or float (H,W,3)
            The colour data to which the perturbation is to be applied
               
        Returns
        -------
        data_out : ndarray of int or float (H,W,3)
            The colour data after the perturbation is applied
        """
        
        #Compute the displacement vector
        _disp_vec = np.matmul(self._princomps, self._disp_values)
        _disp_vec = np.tile(_disp_vec, (self._data.shape[0], self._data.shape[1], 1))
        
        #Apply the displacement vector
        data_out = np.add(self._data, _disp_vec)    
        
        return data_out
        
class PCAColourAugmenter(ColourPreprocessor):
    """PCA-based colour augmentation as described in [1] but with a deterministic
    approach 

    References
    ----------
    [1] Krizhevsky, A., Sutskever, I., Hinton, G.E.
        ImageNet classification with deep convolutional neural networks
        (2017) Communications of the ACM, 60 (6), pp. 84-90.
    """
    
    def __init__(self, perturbation_type, **kwargs):
        """Constructor

        Parameters
        ----------
        perturbation_type : str 
            The type of perturbation to be applied. Can be:
                'translational' -> translation along the principal axes of the
                                   colour distribution
        
        Optional parameters for perturbation_type = 'translational'
        disp_values : 
            The value of the displacements as a fraction of the radii of gyration
            of the corresponding principal axes. For example, let rI, rII and rIII 
            indicate the radii of gyration along the 1st, 2nd a 3rd principal axes.
            Values = (0.1, 0.3, -0,5) will indicate a translation of 
            0.1*r1 along I, 0.3*r2 along II and -0.5*r3 along III 
        """
          
        #Call the superclass constructor
        super().__init__()
        
        #Set the perturbation type
        self.perturbation_type = perturbation_type
        self.kwargs = kwargs
                                                                   
    def _preprocess(self):
        
        #Define the perturbation
        self._perturbation = None
        if self.perturbation_type == 'translational':
            disp_values = self.kwargs['disp_values']
            self._perturbation = PCATranslationalPerturbation(self._img_in, disp_values)
        else:
            raise Exception('Perturbation type not supported')        
                
        #Apply the perturbation
        _perturbed_data = self._perturbation.apply()
                    
        #Append to output
        _img_out = deepcopy(self._img_in)
        _img_out.set_data(_perturbed_data, round_and_window=True)
        self._preprocessed_images.append(_img_out) 



        