import numpy as np
import matplotlib.pyplot as plt
import cenotaph.colour_preprocessing.colour_constancy as cc
from cenotaph.basics.base_classes import Image

img_file = '../images/peppers.jpg'
img_in = Image(img_file)

orig_fig = plt.figure()
plt.imshow(img_in.get_data())
plt.title('Original image')

grey_world = cc.GreyWorld()
img_out = grey_world.get_result(img_in)[0]

gw_fig = plt.figure()
plt.title('Colour pre-processed image -- grey world')
plt.imshow(img_out.get_data())

max_white = cc.MaxWhite()
img_out = max_white.get_result(img_in)[0]

maxw_fig = plt.figure()
plt.imshow(img_out.get_data())
plt.title('Colour pre-processed image -- max white')

stretch = cc.Stretch()
img_out = stretch.get_result(img_in)[0]

str_fig = plt.figure()
plt.imshow(img_out.get_data())
plt.title('Colour pre-processed image -- stretch')

chroma = cc.Chroma()
img_out = chroma.get_result(img_in)[0]

str_fig = plt.figure()
plt.imshow(img_out.get_data())
plt.title('Colour pre-processed image -- chroma')

#Show the original image and the colour-normalised ones
plt.show()
