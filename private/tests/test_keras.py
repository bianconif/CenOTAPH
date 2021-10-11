#from keras.applications.vgg16 import VGG16 as keras_vgg16
#from keras.preprocessing import image

#from keras.applications import MobileNet  as keras_mobilenet
#from keras.applications.resnet50 import ResNet50 as keras_resnet50

#test_img = '../../cenotaph/images/peppers.jpg'

##base_model = keras_vgg16(weights='imagenet', include_top=True)
##img = image.load_img(test_img, target_size=(224, 224))
##base_model.summary()

##base_model = keras_mobilenet(weights='imagenet', include_top=True)
##img = image.load_img(test_img, target_size=(224, 224))
##base_model.summary()

#base_model = keras_resnet50(weights='imagenet', include_top=True)
#img = image.load_img(test_img, target_size=(224, 224))
#base_model.summary()
#a = 0

from keras import layers
from keras import models
from keras.initializers import Zeros

fltr_shape = (5,5)
nfilters = 25
input_shape = (None, 100, 100, 1)
kernel_initializer = Zeros()

print("Building Model...")
model = models.Sequential()
model.add(layers.Conv2D(nfilters, fltr_shape, kernel_initializer=kernel_initializer))

print("Weights before change:")
print (model.layers[0].get_weights())

inp = Input(shape=(100,100,1))
output   = Convolution2D(1, 3, 3, border_mode='same', init='normal',bias=False)(inp)
model_network = Model(input=inp, output=output)

w = np.asarray([ 
    [[[
    [0,0,0],
    [0,2,0],
    [0,0,0]
    ]]]
    ])
input_mat = np.asarray([ 
    [[
    [1.,2.,3.],
    [4.,5.,6.],
    [7.,8.,9.]
    ]]
    ])
model_network.layers[1].set_weights(w)
print("Weights after change:")
print(model_network.layers[1].get_weights())
print("Input:")
print(input_mat)
print("Output:")
print(model_network.predict(input_mat))