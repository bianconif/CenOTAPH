import numpy as np
import cenotaph.filtering as flt
import matplotlib.pyplot as plt
import cenotaph.basics.visualisation as vs
from cenotaph.basics.base_classes import Image

#filter_1 = np.asarray([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
#filter_2 = np.asarray([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])
#filter_1 = np.expand_dims(filter_1, 2)
#filter_2 = np.expand_dims(filter_2, 2)
#filter_bank = np.concatenate((filter_1,filter_2),2)

#filter_bank = flt.DCT()
#filter_bank = flt.Laws()

#num_scales = 5
#num_orientations = 5
#size = 5
#filter_bank = flt.Gabor(size = size, scales = num_scales, 
                        #orientations = num_orientations)
                        
filter_bank = flt.Zernike()

feature_type = 'M'

imgfile = '../../cenotaph/images/peppers.jpg'
img_in = Image(imgfile)
t = filter_bank.get_transformed_images(img_in)

#Show the original image
orig_img = plt.figure()
plt.imshow(img_in.get_data())
plt.title('Original image')

#Show the filters
filters = filter_bank.get_filter_bank()
num_filters = filters.shape[2]
filters_to_show = []
for n in np.arange(num_filters):
    filters_to_show.append(filters[:,:,n])
    
vs.show_images(filters_to_show, cols = 5, titles = None) 


#Show the transformed images
images_to_show = []
for n in np.arange(num_filters):
    images_to_show.append(t[:,:,n])

vs.show_images(images_to_show, cols = 5, titles = None)    

plt.show()

#Get the features
features = f.get_features(img_in)
print('The features are: ' + str(features))