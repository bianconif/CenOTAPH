from cenotaph.third_parties.doc_inherit import doc_inherit

from cenotaph.texture.hep.basics import HEP, HEPLocalThresholding 
from cenotaph.texture.hep.greyscale import LBP
from cenotaph.basics.matrix_displaced_copies import matrix_displaced_copies

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
               
class OCLBP(HEPColour, HEPLocalThresholding):
    """Opponent-colour Local binary patterns"""
    
    @doc_inherit
    def _get_pivot(self):
        
        pivots = list()
        
        #Intra-channel pivots
        for channel in self._colour_layers:
            pivots.append(channel[:,:,self._neighbourhood.center_index()])
        
        #Inter-channel pivots
        pivots.append(self._colour_layers[1][:,:,self._neighbourhood.center_index()])
        pivots.append(self._colour_layers[1][:,:,self._neighbourhood.center_index()])
        pivots.append(self._colour_layers[2][:,:,self._neighbourhood.center_index()])
    
        return pivots

    @doc_inherit
    def _get_base_values(self):
        
        base_values = list()
        
        #Intra-channel base values
        for channel in self._colour_layers:
            base_values.append(channel[:,:,self._neighbourhood.peripheral_indices()])  
            
        #Inter-channel base values -- respectively R, G and B
        base_values.append(self._colour_layers[0][:,:,self._neighbourhood.peripheral_indices()])
        base_values.append(self._colour_layers[1][:,:,self._neighbourhood.peripheral_indices()])
        base_values.append(self._colour_layers[2][:,:,self._neighbourhood.peripheral_indices()])        
        
        
        return base_values
    
    #------------------- Partial inheritance from LBP-----------------------                    
    _get_dictionary = LBP.__dict__['_get_dictionary']
    _get_thresholds = LBP.__dict__['_get_thresholds']
    _consider_equalities = LBP.__dict__['_consider_equalities']
    _get_num_colours = LBP.__dict__['_get_num_colours']
    #----------------------------------------------------------------------