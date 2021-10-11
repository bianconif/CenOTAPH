from abc import abstractmethod
import os

#from keras_segmentation.models.unet import vgg_unet, unet_mini
#from keras_segmentation.pretrained import pspnet_50_ADE_20K,\
     #pspnet_101_cityscapes, resnet_pspnet_VOC12_v0_1
     
from keras.optimizers import Adam
from keras.models import model_from_json

import segmentation_models as sm

import numpy as np

import pickle

import random

import tensorflow.keras.backend as K

from cenotaph.basics.base_classes import Image, ImageType
from cenotaph.basics.generic_functions import clear_folder, get_files_in_folder
from cenotaph.preprocessing.space.crop_and_split import Resize
from cenotaph.preprocessing.segmentation.hand_designed import ImageSegmenter
from cenotaph.third_parties.doc_inherit import doc_inherit

class CNN(ImageSegmenter):
    """Base class for segmenters based on CNN models. The approach is to
    maintain the native field-of-view of the backbone models and to resize
    the images to segment. Works for both greyscale and colour images."""
    
    def __repr__(self):
        fov = self.get_fov()
        return self.__class__.__name__ + f'--fov-{fov[0]}x{fov[1]}' 
    
    #Override
    def get_short_name(self):
        """Short representation
        
        Returns
        -------
        short_name : str
        """
        return self.__repr__()    
    
    def __init__(self, fov, activation = 'softmax', encoder_weights = 'imagenet', 
                 resize_method = 'bicubic', spool = './'):
        """
        Parameters
        ----------
        fov : int (H,W)
            The input size (field of view of the network). Use 'auto' to set as
            input size the native field-of-view of the backbone model.
        activation : str
            The activation function. Possible values are:
                'softmax'
                'sigmoid'
        encoder_weights : str
            Type of pre-trained weights. Possible values are:
                'imagenet' -> ImageNet pre-trained weights
                None       -> Random weights
        resize_method : str
            The method used for resizing the input images to fit the field of
            view of the network backbone. Possible values are:
                'nearest'  -> Nearest neighbour interpolation
                'lanczos'  -> Lanczos interpolation 
                'bilinear' -> Bilinear interpolation 
                'bicubic'  -> Bicubic interpolation
        spool : str
            Path to the folder where to store the temporary images.
        """
        if fov == 'auto':
            self._fov = self.get_native_fov()[0:2]
        else:
            self._fov = fov
        self._activation = activation
        self._encoder_weights = encoder_weights
        self._spool = spool
                
        #Create the resizer for scaling the input images to match the field of
        #view of the network
        fov = self.get_fov()
        self._resizer = Resize(new_size = fov, 
                               interp = 'bicubic')
        
        #Flag the model as not trained
        self._trained = False    
    
    def get_fov(self):
        """Return the field of view of the network
        
        Returns
        -------
        fov : int (W,H)
            The field of view of the network.
        """
        return self._fov
    
    @abstractmethod  
    def get_native_fov(self):
        """Return the native field of view of the network backbone.
        
        Returns
        -------
        fov : int (W,H)
            The native field of view of the network backbone.
        """      
    
    @abstractmethod 
    def get_backbone_name(self):
        """Return the name of the backbone network
        
        Returns
        -------
        backbone_name : str
            The name of the backbone network.
        """      
    
    @abstractmethod
    def _get_model_constructor(self):
        """Return the model constructor
        
        Returns
        -------
        model_constructor: tensorflow.python.keras.engine.functional.Functional
            The model contstructor to be invoked for creating the model
        """
                
    @abstractmethod    
    def _create_model(self):
        """Abstract method for model creation. Each subclass needs to 
        implement the creation of the specific model.
        
        Returns
        -------
        model : tensorflow.python.keras.engine.functional.Functional
            The segmentation model.
        """
        
    def save(self, out_file):
        """Save the model
        
        Parameters
        ----------
        out_file : str
            Path to the destination file (.h5)
        """

        self._model.save(out_file)
                
    def _check_model(self, in_file):
        """Check that the loaded model is compatible with the current one
        
        Parameters
        ----------
        in_file : str
            Path to the stored model (.h5).
            
        Returns
        -------
        compatible : bool
            Flag indicating whether the stored model is compatible with the 
            current one.
        """
        compatible = True
        
        #Create a dummy model
        dummy = self._create_model()
        
        #Load the stored model
        dummy.load_weights(in_file)
        
        #Do the check
        if not (dummy.n_classes == self._num_classes):
            compatible = False
        if not (dummy.model_name == self.get_inner_name()):
            compatible = False
            
        return compatible
            
    
    @doc_inherit    
    def _segment(self, img, **kwargs):
        
        #Store the original size
        H, W = img.get_size()[0:2]
        
        #If the image is single-channel convert it to three channels
        img = img.get_multichannel()
                
        #Resize the input image to match the field of view of the network
        resized_img = self._resizer.get_result(img)
        
        #Convert to float and scale to [0,1] 
        network_feed = resized_img.normalised_as_float()
                
        #Do the segmentation
        network_feed = np.expand_dims(network_feed, axis = 0)
        network_raw_output = self._model.predict(x = network_feed)
        network_rounded_output = np.rint(network_raw_output).astype(np.uint8)
        
        #Convert back from one-hot encoding // assume there are at most 256 
        #classes
        if network_rounded_output.shape[3] > 1:
            out = np.zeros((network_rounded_output.shape[1], 
                            network_rounded_output.shape[2]))
            for i in range(network_rounded_output.shape[3]):
                slice_ = network_rounded_output[0,:,:,i]
                slice_ = np.squeeze(slice_)
                out[np.where(slice_ == i)] = i
        else:
            out = np.squeeze(network_rounded_output)
                       
        segmented = Image.from_array(data = out, img_type = ImageType.GS)
        
        #Scale back the result to the size of the input image
        back_resizer = Resize(new_size = (H, W), interp = 'nearest')
        segmented_resized = back_resizer.get_result(segmented)
        
        return segmented_resized.get_data()
        
class TrainableCNN(CNN):
    """Base class for trainable CNN"""
    
    def __repr__(self):
        if self._encoder_frozen:
            mode = 'fine-tuned'
        else:
            mode = 'full-trained'
        return super().__repr__() + f'--{mode}'     
    
    def __init__(self, num_classes, fov='auto', activation = 'softmax',
                 encoder_weights = 'imagenet', encoder_frozen = False,
                 resize_method = 'bicubic', spool = './'):
        """
        Parameters
        ----------
        num_classes : int
            The number of classes.
        fov : int (H,W)
            The input size (field of view of the network). Use 'auto' to set as
            input size the native field-of-view of the backbone model.
        activation : str
            The activation function. Possible values are:
                'softmax'
                'sigmoid'
        encoder_weights : str
            Type of pre-trained weights. Possible values are:
                'imagenet' -> ImageNet pre-trained weights
                None       -> Random weights
        encoder_frozen : bool
            If True the encoder's weight are frozen during training (only
            the decoder is trained).
        resize_method : str
            The method used for resizing the input images to fit the field of
            view of the network backbone. Possible values are:
                'nearest'  -> Nearest neighbour interpolation
                'lanczos'  -> Lanczos interpolation 
                'bilinear' -> Bilinear interpolation 
                'bicubic'  -> Bicubic interpolation
        spool : str
            Path to the folder where to store the temporary images.
        """
        super().__init__(fov = fov,
                         activation = activation, 
                         encoder_weights = encoder_weights, 
                         resize_method = resize_method, 
                         spool = spool)
        
        self._encoder_frozen = encoder_frozen
        self._num_classes = num_classes
        
        #Create the model
        self._create_model()
       
    def _create_model(self):
        """Create the segmentation model"""
        constructor = self._get_model_constructor()
        fov = self.get_fov()
        self._model = constructor(input_shape = (fov[0], fov[1], 3),
                                  backbone_name = self.get_backbone_name(),
                                  activation = self._activation,
                                  classes = self._num_classes,
                                  encoder_weights = self._encoder_weights,
                                  encoder_freeze = self._encoder_frozen)       
        
    def _generate_data(self, img_folder, mask_folder, batch_size, class_labels):
        """The data generator for feeding the net during training
        
        Parameters
        ----------
        img_folder : str
            Path to the folder containing the images.
        mask_folder : str
            Path to the folder containing the annotations.
        class_labels : list of int
            The class labels used in the masks. There must be as many labels
            as num_classes.
        batch_size : int
            The batct size.
            
        References
        ----------
        [1] Chakraborty, R. A Keras Pipeline for Image Segmentation
            https://towardsdatascience.com/a-keras-pipeline-for-image-
            segmentation-part-1-6515a421157d
            Visited on 1 Nov. 2020
        """
        c = 0
        images = get_files_in_folder(img_folder) 
        masks = get_files_in_folder(mask_folder)
        
        #Make sure there are as many images as masks
        if len(images) != len(masks):
            raise Exception('There should be as many images as masks')
        
        indices = np.arange(len(images))
        random.shuffle(indices)      
        fov = self.get_fov()
        
        while (True):
            img = np.zeros((batch_size, fov[0], fov[1], 3)).astype('float')
            mask = np.zeros((batch_size, fov[0], fov[1], len(class_labels))).astype('float')
        
            for i in range(c, c+batch_size): #initially from 0 to 16, c = 0. 
                
                #Read image from folder, resize and normalise to [0,1]
                train_img = Image(images[indices[i]])
                train_img = self._resizer.get_result(train_img)
                
                #Add the image to the batch. If the image is not three channel
                #(greyscale) compy the intensity channel to each of the R, G and
                #B channels
                if train_img.get_n_channels() < 3:
                    train_img = train_img.get_multichannel()   
                data = train_img.normalised_as_float()
                img[i-c] = data 
        
                #Read the corresponding mask
                train_mask = Image(masks[indices[i]])
                train_mask = self._resizer.get_result(train_mask)
                
                #Add the mask to the batch -- not sure normalisation is needed 
                #here, need to check
                #data = train_mask.normalised_as_float()
                labels = train_mask.get_one_hot(lut = class_labels)
                if labels.ndim < 3:
                    labels = np.expand_dims(a = labels, axis = -1)                 
                mask[i-c] = labels
        
            c+=batch_size
            if(c+batch_size>=len(images)):
                c = 0
                random.shuffle(indices)

            yield img, mask    
    
    def _load_model(self, in_dir):
        """Load the model from a folder. 
        
        Parameters
        ----------
        out_dir : str
            Path to the folder where to store the model.
            
        Returns
        -------
        load_ok : bool
            Flag indicating that the model was loaded correctly.
        """
        
        load_ok = True
        
        try:           
            #Load the model from JSON file
            with open(in_dir + '/model.json', 'r') as json_file:
                self._model = model_from_json(json_file.read())
            
            #Load the weights
            self._model.load_weights(in_dir + '/weights.h5') 
        except:
            load_ok = False
            
        return load_ok
    
    def _save_model(self, out_dir):
        """Save the model into a folder.
        
        Parameters
        ----------
        out_dir : str
            Path to the folder where to store the model.
        """
        #Serialize model to JSON
        with open(out_dir + '/model.json', 'w') as json_file:
            json_file.write(self._model.to_json())
        
        #Serialize weights to HDF5
        self._model.save_weights(out_dir + '/weights.h5')
         
    
    def train_model(self, num_epochs, steps_per_epoch, validation_steps,
                    validation_freq, train_images, train_annotations, 
                    loss_function, metrics_to_monitor, model_cache, 
                    class_labels, callbacks = None, train_batch_size=1, 
                    val_batch_size=1, val_images=None, val_annotations=None, 
                    run_eagerly=False):
        """Train a segmentation model and cache it into a file. If the file
        already exists just load it without training again.
            
        Parameters
        ----------
        num_epochs : int
            The number of epochs used for training the model.
        steps_per_epoch : int
            The number of epochs for each step.
        validation_steps : int
            Total number of steps (batches of samples) to draw during 
            validation. Has no effect is val_images = None.
        validation_freq : int
            Run validation every n epochs. Has no effect is val_images = None.
        train_images : str
            Path to the folder containing the train images.
        train_annotations : str
            Path to the folder containing the pre-segmented images for training
            (ground truth).  
        loss_function : function
            The The loss function used for optimising the model. 
        metrics_to_monitor : list of functions
            The metrics used for monitoring the performance of the model during
            training.
        model_cache : str
            Pointer to the folder where the trained model is to be stored.
            The model will be saved into two files: 'model.json' (structure)
            and 'weights.h5' (the trained weights).
        class_labels : list of int
            The class labels used in the masks. There must be as many labels
            as num_classes.
        callbacks : List of keras.callbacks.Callback instances. 
            List of callbacks to apply during training. See tf.keras.callbacks.
        train_batch_size : int
            Number of images considered during each train step.
        val_batch_size : int
            Number of images considered during each validation step.
        val_images : str
            Path to the folder containing the validation images. Pass None if
            no validation is required.
        val_annotations : str
            Path to the folder containing the pre-segmented images for
            validation (ground truth). Pass None if no validation is required.
            Labelling follows the same convention as train_annotations.
        run_eagerly : bool
            Use true to allow debugging of Tensorflow functions.
        """
        
        #Make sure the class labels given are compatible with the number of
        #classes
        if len(class_labels) != self._num_classes:
            raise Exception('There must be as many labels as classes')
        
        if self._load_model(model_cache):
            #Model loaded correctly, no other operation needed  
            pass
        else:
            #Train the model
            
            #Define the optimiser
            optimizer = Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.999, 
                             epsilon=1e-7, amsgrad=False, name='Adam')
            
            #Define the generators for train and validation data
            train_gen = self._generate_data(img_folder = train_images, 
                                            mask_folder = train_annotations, 
                                            batch_size = train_batch_size,
                                            class_labels = class_labels)
            val_gen = None
            if val_images is not None:
                val_gen = self._generate_data(img_folder = val_images, 
                                              mask_folder = val_annotations, 
                                              batch_size = val_batch_size,
                                              class_labels = class_labels)            
            
            #Compile the model
            self._model.compile(optimizer = optimizer, 
                                loss = loss_function,
                                metrics = metrics_to_monitor, 
                                loss_weights = None, 
                                weighted_metrics = None,
                                run_eagerly = run_eagerly)
            
            #Train the model
            history = self._model.fit(
                x = train_gen,
                y = None,
                batch_size = None,
                epochs = num_epochs,
                verbose = 1,
                callbacks = callbacks,
                validation_split=0.0,
                validation_data = val_gen,
                shuffle = None,
                class_weight = None,
                sample_weight = None,
                initial_epoch = 0,
                steps_per_epoch = steps_per_epoch,
                validation_steps = validation_steps,
                validation_batch_size = val_batch_size,
                validation_freq = validation_freq,
                max_queue_size = 10,
                workers=1,
                use_multiprocessing=False)
            
            #Flag the model as trained
            self._trained = True            
                        
            #Save the train history
            with open(model_cache + '/history.pkl', 'wb') as fp_history:
                pickle.dump(history.history, fp_history)
                        
            #Store the trained model
            self._save_model(model_cache)    

#***************************************************
#******************* Encoders **********************
#***************************************************
class Encoder():
    """Abstract base class for CNN encoder"""
    
    @abstractmethod
    def get_native_fov(self):  
        """To be implemented by the subclasses"""
    
    @abstractmethod
    def get_backbone_name(self):
        """To be implemented by the subclasses"""
        
class InceptionV3(Encoder):
          
    @doc_inherit
    def get_native_fov(self):
        #Native value = (299, 299, 3); closest multiple of 32 (288, 288, 3)
        return (288, 288, 3)   
            
    @doc_inherit
    def get_backbone_name(self):
        return 'inceptionv3'
        
class MobileNet(Encoder):
          
    @doc_inherit
    def get_native_fov(self):
        return (224, 224, 3)   
            
    @doc_inherit
    def get_backbone_name(self):
        return 'mobilenet'

class ResNet34(Encoder):
    @doc_inherit
    def get_native_fov(self):
        return (224, 224, 3)   
    
    @doc_inherit
    def get_backbone_name(self):
        return 'resnet34'        
    

#***************************************************
#************* Segementation models ****************
#***************************************************
class SegmentationModel():
    """Abstract base class for Segmentation model"""
    
    @abstractmethod
    def _get_model_constructor(self):
        """To be implemented by the subclasses"""
  
class FPN(SegmentationModel):
            
    @doc_inherit
    def _get_model_constructor(self):
        return sm.FPN
    
class LinkNet(SegmentationModel):
            
    @doc_inherit
    def _get_model_constructor(self):
        return sm.Linknet
    
class PSPNet(SegmentationModel):
            
    @doc_inherit
    def _get_model_constructor(self):
        return sm.PSPNet
        
class Unet(SegmentationModel):
            
    @doc_inherit
    def _get_model_constructor(self):
        return sm.Unet
    

        
#***************************************************
#******** Segementation models + encoders **********
#***************************************************
class FPN_MobileNet(FPN, MobileNet, TrainableCNN):
    pass

class FPN_InceptionV3(FPN, InceptionV3, TrainableCNN):
    pass

class FPN_ResNet34(FPN, ResNet34, TrainableCNN):
    pass

class LinkNet_MobileNet(LinkNet, MobileNet, TrainableCNN):
    pass

class LinkNet_InceptionV3(LinkNet, InceptionV3, TrainableCNN):
    pass

class LinkNet_ResNet34(LinkNet, ResNet34, TrainableCNN):
    pass

class Unet_MobileNet(Unet, MobileNet, TrainableCNN):
    pass

class Unet_InceptionV3(Unet, InceptionV3, TrainableCNN):
    pass

class Unet_ResNet34(Unet, ResNet34, TrainableCNN):
    pass

class PSPNet_MobileNet(PSPNet, MobileNet, TrainableCNN):
    
    @doc_inherit
    #Override -- input needs to be multiple of 48
    def get_native_fov(self):
        return (240, 240, 3)  

class PSPNet_InceptionV3(PSPNet, InceptionV3, TrainableCNN):
    pass

class PSPNet_ResNet34(PSPNet, ResNet34, TrainableCNN):
    
    @doc_inherit
    #Override -- input needs to be multiple of 48
    def get_native_fov(self):
        return (240, 240, 3) 

 
    

    

        
    
  
    
        