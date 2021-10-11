import numpy as np
import matplotlib.pyplot as plt

import cenotaph.preprocessing.space.crop_and_split as cas
from cenotaph.basics.base_classes import Image

img_file = '../../cenotaph/images/peppers.jpg'
#img_file = '../../cenotaph/images/landscape-GS-8.jpg'
img_in = Image(img_file)

#***********************************
#*********** Test Resize ***********
#***********************************
new_size = (100,100)
resizer = cas.Resize(new_size, interp='bicubic')
img_out = resizer.get_result(img_in)

orig_fig = plt.figure()
plt.imshow(img_in.get_data())
plt.title('Original image')

cropped_fig = plt.figure()
plt.title('Resized image')
plt.imshow(img_out[0].get_data())
#***********************************
#***********************************
#***********************************

#***********************************
#************ Test Crop ************
#***********************************
#offset = (20,30)
#new_size = (100,100)
#cropper = cas.Crop(offset, new_size)
#img_out = cropper.get_result(img_in)

#orig_fig = plt.figure()
#plt.imshow(img_in.get_data())
#plt.title('Original image')

#cropped_fig = plt.figure()
#plt.title('Cropped image')
#plt.imshow(img_out[0].get_data())
##***********************************
##***********************************
##***********************************

##***********************************
##******** Test CentralCrop *********
##***********************************
#new_size = (200,300)
#ccropper = cas.CentralCrop(new_size)
#img_out = ccropper.get_result(img_in)

#ccropped_fig = plt.figure()
#plt.title('Centrally cropped image')
#plt.imshow(img_out[0].get_data())
##***********************************
##***********************************
##***********************************

##***********************************
##*******  Test UniformSplit ********
##***********************************
#num_splits = (3,2)
#usplitter = cas.UniformSplit(num_splits)
#tiles = usplitter.get_result(img_in)
#count = 1
#for tile in tiles:
    #plt.figure()    
    #plt.imshow(tile.get_data())
    #plt.title('Tile %d of %d' %(count, len(tiles))) 
    #count += 1
##***********************************
##***********************************
##***********************************

orig_fig = plt.figure()
plt.imshow(img_in.get_data())
plt.title('Original image')

#Show the original image and the colour-normalised ones
plt.show()  