import numpy as np

from cenotaph.third_parties.doc_inherit import doc_inherit

from cenotaph.texture.hep.basics import HEP, HEPLocalThresholding
from cenotaph.texture.hep.greyscale import LBP, ILBP, LBPBasics, LBPDict, ILBPDict
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
                   
class OCLBP(ImageDescriptor):
    """Opponent-colour Local binary patterns"""
    
    def __init__(self, radius=1, num_peripheral_points=8, group_action=None,
                 **kwargs):
         
        #Intra-channel calculator
        self._lbp = LBP(radius = radius, 
                        num_peripheral_points = num_peripheral_points, 
                        group_action = group_action,
                        **kwargs) 
        self._intra_channel_lbp = IntraChannelImageDescriptor(self._lbp)
        
        #Inter-channel calculator
        self._inter_channel_lbp = InterChannelLBP(
            radius = radius, num_peripheral_points = num_peripheral_points, 
            group_action = group_action, **kwargs)
        
    def _compute_features(self):
        
        intra_channel_features = self._intra_channel_lbp.get_features(
            self._img_in) 
        inter_channel_features = self._inter_channel_lbp.get_features(
            self._img_in)
        
        return np.hstack((intra_channel_features, inter_channel_features))
                
    
class IOCLBP(ImageDescriptor):
    """Opponent-colour Local binary patterns"""
    
    def __init__(self, radius=1, num_peripheral_points=8, group_action=None,
                 **kwargs):
         
        #Intra-channel calculator
        self._ilbp = ILBP(radius = radius, 
                          num_peripheral_points = num_peripheral_points, 
                          group_action = group_action,
                          **kwargs) 
        self._intra_channel_ilbp = IntraChannelImageDescriptor(self._ilbp)
        
        #Inter-channel calculator
        self._inter_channel_ilbp = InterChannelILBP(
            radius = radius, num_peripheral_points = num_peripheral_points, 
            group_action = group_action, **kwargs)
        
    def _compute_features(self):
        
        intra_channel_features = self._intra_channel_ilbp.get_features(
            self._img_in) 
        inter_channel_features = self._inter_channel_ilbp.get_features(
            self._img_in)
        
        return np.hstack((intra_channel_features, inter_channel_features))