from cenotaph.basics.base_classes import Image
import cenotaph.cnn as cnn

img_file = '../../cenotaph/images/peppers.jpg'
img_in = Image(img_file)

densenet121 = cnn.DenseNet121()
f = vgg16.get_features(img_in)
