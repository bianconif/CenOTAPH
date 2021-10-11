from abc import abstractmethod

from keras.models import Model
from keras.applications import DenseNet121 as keras_densenet121
from keras.applications import MobileNet as keras_mobilenet
from keras.applications.imagenet_utils import preprocess_input\
     as keras_preprocess_input
from keras.applications.resnet50 import ResNet50 as keras_resnet50
from keras.applications.resnet50 import preprocess_input as\
     keras_resnet50_preprocess_input
from keras.applications.vgg16 import VGG16 as keras_vgg16
from keras.applications.vgg16 import preprocess_input as\
     keras_vgg16_preprocess_input
from keras.applications import Xception as keras_xception

import numpy as np

from cenotaph.basics.base_classes import ImageDescriptor
from cenotaph.basics.base_classes import ImageType
from cenotaph.third_parties.doc_inherit import doc_inherit
from cenotaph.preprocessing.space.crop_and_split import Resize

class PreTrainedCNN(ImageDescriptor):
    """Wrapper class for generic pre-trained model based on Keras"""
       
    #Accepted values for input
    _valid_weights = {'imagenet'}
    _valid_layers = {'next-to-last'}   
    
    @abstractmethod
    def __init__(self, model, weights='imagenet', include_top=True,
                 layer='next-to-last'):
        """Default constructor
        
        Parameters
        ----------
        model : keras.engine.training.Model
            The pretrained model. See keras documentaion for possible values.
        weights : str
            The checkpoint from which to initialise the model. Can be:
                'imagenet' -> Network trained on the imagenet dataset.
        include_top : bool
            Whether to include the fully-connected layers at the top of tge
            network
        layer : str
            The layer from which the features are extracted. Can be:
                'next-to-last' -> last fully-connected layer
        """

        
        if weights not in self._valid_weights:
            raise Exception('Weights checkpoint not supported')
        if layer not in self._valid_layers:
            raise Exception('Layer not supported')
         
        self._weights = weights
        self._include_top = include_top
        self._layer = layer
                
        #Define the layer from which the output is taken
        layer_m = self._map_layer_name(self._layer)
        
        #Create the model based on that layer
        self._model = Model(inputs=self._base_model.input, 
                            outputs=self._base_model.get_layer(layer_m).output)        
    
    @doc_inherit        
    def get_features(self, img):
                
        #If the input is single-channel create a repeated image
        if img.get_n_channels() == 1:
            num_net_channels = self._model.get_input_shape_at(0)[3]
            img = img.get_multichannel(num_net_channels)
        
        return super().get_features(img)
    
    def get_settings(self):
        #Return a string with the settings
        return self._weights + '-' + self._layer
    
    def __repr__(self):
        return super().__repr__() + '-' + self.get_settings()
    
    @abstractmethod 
    def _map_layer_name(self, layer_w):
        """Map the name of the layer in the wrapper (self._layer) to
        the name of the layer in the inner model
        
        Parameters
        ----------
        layer_w : str
            The name of the layer in the wrapper (stored as self._layer)
        
        Returns
        -------
        layer_m : str
            The name of the layer in the inner model
        """       

    @doc_inherit
    def _compute_features(self):
                
        #Resize the input image
        resizer = Resize(self._default_input_size)
        resized_img = resizer.get_result(self._img_in)
        
        #Prepare the input to feed the net
        x = resized_img.get_data()
        x = np.expand_dims(x, axis=0)
        x = self._preprocess_input(x) 
        
        #Compute the features and flatten the result
        features = self._model.predict(x) 
        features = np.ndarray.flatten(features)
        
        return features
                
class DenseNet121(PreTrainedCNN):
    
    #Default input size
    _default_input_size = (224, 224)
        
    @doc_inherit
    def __init__(self, weights='imagenet', include_top=True, 
                 layer='next-to-last'):
        
        #Load the base model
        self._base_model = keras_densenet121(include_top = include_top, 
                                             weights = weights)
        
        #Set the pre-processing function
        self._preprocess_input = keras_preprocess_input
        
        #Invoke the superclass constructor
        super().__init__(model = self._base_model, weights = weights, 
                         include_top = include_top, layer = layer)
                   
    @doc_inherit
    def _map_layer_name(self, layer_w):
        layer_m = None
        if layer_w == 'next-to-last':
            layer_m = 'avg_pool'
        else:
            raise Exception('Layer not supported')
        return layer_m    
   
class MobileNet(PreTrainedCNN):
    
    #Default input size
    _default_input_size = (224, 224)
        
    @doc_inherit
    def __init__(self, weights='imagenet', include_top=True, 
                 layer='next-to-last'):
        
        #Load the base model
        self._base_model = keras_mobilenet(include_top=include_top, 
                                           weights=weights)
        
        #Set the pre-processing function
        self._preprocess_input = keras_preprocess_input
        
        #Invoke the superclass constructor
        super().__init__(self._base_model, weights, include_top, layer)
            
    @doc_inherit
    def _map_layer_name(self, layer_w):
        layer_m = None
        if layer_w == 'next-to-last':
            layer_m = 'dropout'
        else:
            raise Exception('Layer not supported')
        return layer_m
   
            
class ResNet50(PreTrainedCNN):
    
    #Default input size
    _default_input_size = (224, 224)
        
    @doc_inherit
    def __init__(self, weights='imagenet', include_top=True, 
                 layer='next-to-last'):
        
        #Load the base model
        self._base_model = keras_resnet50(include_top=include_top, 
                                           weights=weights)
        
        #Set the pre-processing function
        self._preprocess_input = keras_resnet50_preprocess_input
        
        #Invoke the superclass constructor
        super().__init__(self._base_model, weights, include_top, layer)
            
    @doc_inherit
    def _map_layer_name(self, layer_w):
        layer_m = None
        if layer_w == 'next-to-last':
            layer_m = 'avg_pool'
        else:
            raise Exception('Layer not supported')
        return layer_m    

class VGG16(PreTrainedCNN):
    
    #Default input size
    _default_input_size = (224, 224)
        
    @doc_inherit
    def __init__(self, weights='imagenet', include_top=True, 
                 layer='next-to-last'):
        
        #Load the base model
        self._base_model = keras_vgg16(include_top=include_top, 
                                       weights=weights)
        
        #Set the pre-processing function
        self._preprocess_input = keras_vgg16_preprocess_input
        
        #Invoke the superclass constructor
        super().__init__(self._base_model, weights, include_top, layer)
            
    @doc_inherit
    def _map_layer_name(self, layer_w):
        layer_m = None
        if layer_w == 'next-to-last':
            layer_m = 'fc2'
        else:
            raise Exception('Layer not supported')
        return layer_m
        
class Xception(PreTrainedCNN):
    
    #Default input size
    _default_input_size = (299, 299)
        
    @doc_inherit
    def __init__(self, weights='imagenet', include_top=True, 
                 layer='next-to-last'):
        
        #Load the base model
        self._base_model = keras_xception(include_top=include_top, 
                                          weights=weights)
        
        #Set the pre-processing function
        self._preprocess_input = keras_preprocess_input
        
        #Invoke the superclass constructor
        super().__init__(model = self._base_model, weights = weights, 
                         include_top = include_top, layer = layer)
                   
    @doc_inherit
    def _map_layer_name(self, layer_w):
        layer_m = None
        if layer_w == 'next-to-last':
            layer_m = 'avg_pool'
        else:
            raise Exception('Layer not supported')
        return layer_m