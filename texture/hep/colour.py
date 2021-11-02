from abc import abstractmethod

import numpy as np

from cenotaph.third_parties.doc_inherit import doc_inherit

from cenotaph.texture.hep.basics import HEP, HEPLocalThresholding
from cenotaph.texture.hep.greyscale import LBP, ILBP, LBPBasics, LBPDict, ILBPDict
from cenotaph.basics.generic_functions import is_documented_by
from cenotaph.basics.matrix_displaced_copies import matrix_displaced_copies
from cenotaph.basics.base_classes import ImageDescriptor, Ensemble,\
     IntraChannelImageDescriptor

class HEPColour(HEP):
    """Base class for three-channel colour descriptors of the HEP family"""
    
    def _pre_compute_features(self):
        
        #Generate displaced copies of each colour channel
        self._colour_layers = list()
        raw_img_data = self._img_in.get_data()
        for channel in range(raw_img_data.shape[2]):
            self._colour_layers.append(
                matrix_displaced_copies(raw_img_data[:,:,channel], 
                                        self._neighbourhood.\
                                        get_integer_points())) 
            
class InterChannelLBP(LBPBasics, LBPDict, HEPColour, HEPLocalThresholding):
    """Inter-channel LBP features"""
    
    @doc_inherit
    def __init__(self, radius=1, num_peripheral_points=8, group_action=None, 
                 **kwargs):
        super().__init__(radius=radius, 
                         num_peripheral_points=num_peripheral_points, 
                         group_action=group_action, **kwargs)
                
    @doc_inherit
    def _get_pivot(self):
        
        pivots = list()
        
        #Inter-channel pivots
        pivots.append(self._colour_layers[0][:,:,self._neighbourhood.center_index()])
        pivots.append(self._colour_layers[0][:,:,self._neighbourhood.center_index()])
        pivots.append(self._colour_layers[1][:,:,self._neighbourhood.center_index()])
    
        return pivots

    @doc_inherit
    def _get_base_values(self):
        
        base_values = list()
                    
        #Inter-channel base values -- respectively R, G and B
        base_values.append(self._colour_layers[1][:,:,self._neighbourhood.peripheral_indices()])
        base_values.append(self._colour_layers[2][:,:,self._neighbourhood.peripheral_indices()])
        base_values.append(self._colour_layers[2][:,:,self._neighbourhood.peripheral_indices()])        
        
        
        return base_values
    
class InterChannelILBP(LBPBasics, ILBPDict, HEPColour, HEPLocalThresholding):
    """Inter-channel ILBP features"""
    
    @doc_inherit
    def __init__(self, radius=1, num_peripheral_points=8, group_action=None, 
                 **kwargs):
        super().__init__(radius=radius, 
                         num_peripheral_points=num_peripheral_points, 
                         group_action=group_action, **kwargs)
                
    @doc_inherit
    def _get_pivot(self):
        
        pivots = list()
        
        #Inter-channel pivots
        pivots.append(np.mean(self._colour_layers[0], axis = 2))
        pivots.append(np.mean(self._colour_layers[0], axis = 2))
        pivots.append(np.mean(self._colour_layers[1], axis = 2))
    
        return pivots

    @doc_inherit
    def _get_base_values(self):
        
        base_values = list()
                    
        #Inter-channel base values -- respectively R, G and B
        base_values.append(self._colour_layers[1])
        base_values.append(self._colour_layers[2])
        base_values.append(self._colour_layers[2])        
        
        
        return base_values 
    
class _IntraAndInterChannelHEP(ImageDescriptor):
    """Base class for colour HEP descriptors based on intra- and inter-channel
    features"""
    
    @is_documented_by(HEP.__init__)
    def __init__(self, radius=1, num_peripheral_points=8, group_action=None,
                 **kwargs):  
        
        #Store the parameters passed
        self._params = locals().copy()
        self._params.pop('self')
        
        #Set the kernel
        self._set_kernel()
        
    @abstractmethod
    def _set_kernel(self):
        """Set the kernel (intra- and inter-channel) descriptors"""
              
    def _compute_features(self):
        return self._kernel.get_features(self._img_in)
        
    
class OCLBP(_IntraAndInterChannelHEP):
    """Opponent-colour Local binary patterns.
    
    References
    ----------
    [1] Maenpaa, T., Pietikainen, M.
        Texture analysis with local binary patterns
        (2005) Handbook of Pattern Recognition and Computer Vision, 3rd Edition, 
        pp. 197-216. 
    """
    
    def _set_kernel(self):    
        self._kernel = Ensemble([IntraChannelImageDescriptor(LBP(**self._params)), 
                                 InterChannelLBP(**self._params)])
                       
class IOCLBP(_IntraAndInterChannelHEP):
    """Improved Opponent-colour Local binary patterns.
    
    References
    ----------
    [1] Bianconi, F., Bello-Cerezo, R., Napoletano, P.
        Improved opponent color local binary patterns: An effective local image 
        descriptor for color texture classification
        (2018) Journal of Electronic Imaging, 27 (1), art. no. 011002
    """
    
    def _set_kernel(self):    
        self._kernel = Ensemble([IntraChannelImageDescriptor(ILBP(**self._params)), 
                                 InterChannelILBP(**self._params)])