import numpy as np
import matplotlib.pyplot as plt
import cenotaph.colour_preprocessing.colour_constancy as cc
import cenotaph.basics.base_classes as bc
import cenotaph.colour_descriptors as cd
from cenotaph.basics.base_classes import Image

img_file = '../images/peppers.jpg'
img_in = Image(img_file)

#Define the pre-processors
grey_world = cc.GreyWorld()
max_white = cc.MaxWhite()
pre_processors = [grey_world, max_white]

#Define the descriptors
mean = cd.Mean()
descriptors = [mean]

#Set up the pipeline
pipeline = bc.ImageProcessingPipeline(image_descriptors=descriptors,
                                      pre_processors=pre_processors)
#Compute the features
f = pipeline.compute_features(img_in)
print('Original features' + str(f))

#Save the features and load them back
repo = './features.npy'
bc.ImageProcessingPipeline.save_features(f, repo)
f2 = bc.ImageProcessingPipeline.load_features(repo)
print('Features retrieved from file' + str(f))